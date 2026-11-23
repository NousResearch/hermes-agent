//! Read-only model discovery and supported request options for native providers.
use crate::{Config, portal};
use anyhow::{Context, Result, anyhow, ensure};
use reqwest::{Client, Url};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub id: String,
    pub name: String,
    pub context_length: Option<u64>,
    pub max_output_tokens: Option<u64>,
    #[serde(default)]
    pub supported_parameters: Vec<String>,
    #[serde(default)]
    pub free: bool,
}
impl Model {
    pub fn supports(&self, parameter: &str) -> bool {
        self.supported_parameters
            .iter()
            .any(|name| name == parameter)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationOptions {
    pub temperature: Option<f64>,
    pub max_output_tokens: Option<u32>,
    pub reasoning_effort: Option<String>,
    #[serde(default)] pub fast:bool,
    #[serde(default)]
    pub reasoning_object: bool,
    #[serde(default)]
    pub max_completion_tokens: bool,
}
impl GenerationOptions {
    pub fn validate(&self) -> Result<()> {
        if let Some(value) = self.temperature {
            ensure!(
                value.is_finite() && (0.0..=2.0).contains(&value),
                "Temperature must be between 0 and 2"
            );
        }
        if let Some(value) = self.max_output_tokens {
            ensure!(
                (1..=1_048_576).contains(&value),
                "Maximum output tokens must be between 1 and 1,048,576"
            );
        }
        if let Some(value) = &self.reasoning_effort {
            ensure!(
                ["none", "minimal", "low", "medium", "high", "xhigh","max","ultra"].contains(&value.as_str()),
                "Unsupported reasoning effort"
            );
        }
        Ok(())
    }
    pub(crate) fn apply(&self, body: &mut Value, anthropic: bool) -> Result<()> {
        self.validate()?;
        if self.fast && !anthropic{body["service_tier"]=json!("priority");}
        if let Some(value) = self.temperature {
            if anthropic {
                ensure!(
                    value <= 1.0,
                    "Anthropic temperature must be between 0 and 1"
                );
            }
            body["temperature"] = json!(value);
        }
        if let Some(value) = self.max_output_tokens {
            body[if !anthropic && self.max_completion_tokens {
                "max_completion_tokens"
            } else {
                "max_tokens"
            }] = json!(value);
        }
        if let Some(value) = &self.reasoning_effort {
            ensure!(
                !anthropic,
                "Native Anthropic reasoning configuration is not supported by this model control"
            );
            let value=if value=="ultra"{"max"}else{value.as_str()};
            if self.reasoning_object && value=="none"{body["reasoning"]=json!({"enabled":false});}
            else if self.reasoning_object {
                body["reasoning"] = json!({"effort":value});
            } else {
                body["reasoning_effort"] = json!(value);
            }
        }
        Ok(())
    }
}

/// GET /models only: selecting or inspecting a model never invokes inference.
pub async fn discover(config: Config) -> Result<Vec<Model>> {
    let mut config = config;
    if config.model.trim().is_empty() {
        config.model = "catalog-discovery".into();
    }
    config.validate()?;
    let config = portal::catalog_credentials(config).await?;
    let mut url = Url::parse(config.base_url.trim()).context("Invalid model catalog endpoint")?;
    let path = url.path().trim_end_matches('/');
    let anthropic = url.host_str() == Some("api.anthropic.com") || path.ends_with("/messages");
    let base = path
        .strip_suffix("/chat/completions")
        .or_else(|| path.strip_suffix("/messages"))
        .unwrap_or(path);
    let path = format!("{}/models", if base.is_empty() { "/v1" } else { base });
    url.set_path(&path);
    let client = Client::builder()
        .user_agent(concat!("HermesRust/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(Duration::from_secs(15))
        .timeout(Duration::from_secs(35))
        .redirect(reqwest::redirect::Policy::none())
        .build()?;
    let mut request = client.get(url).header("accept", "application/json");
    if anthropic {
        request = request.header("anthropic-version", "2023-06-01");
        if !config.api_key.trim().is_empty() {
            request = request.header("x-api-key", config.api_key.trim());
        }
    } else if !config.api_key.trim().is_empty() {
        request = request.bearer_auth(config.api_key.trim());
    }
    let mut response = request
        .send()
        .await
        .map_err(|_| anyhow!("Could not reach the provider model catalog"))?;
    ensure!(
        response.status().is_success(),
        "Model discovery returned HTTP {}; check this provider's connection and credentials",
        response.status()
    );
    let mut bytes = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|_| anyhow!("Model catalog download was interrupted"))?
    {
        ensure!(
            bytes.len() + chunk.len() <= 16 * 1_048_576,
            "Model catalog exceeded the maximum size"
        );
        bytes.extend_from_slice(&chunk);
    }
    let value: Value =
        serde_json::from_slice(&bytes).context("Provider returned an invalid model catalog")?;
    let entries = value
        .get("data")
        .and_then(Value::as_array)
        .or_else(|| value.get("models").and_then(Value::as_array))
        .context("Provider returned no model list")?;
    let mut models = Vec::new();
    for entry in entries.iter().take(20_000) {
        let Some(id) = entry
            .get("id")
            .or_else(|| entry.get("name"))
            .and_then(Value::as_str)
            .filter(|id| !id.trim().is_empty() && id.len() <= 1024)
        else {
            continue;
        };
        let supported_parameters = entry
            .get("supported_parameters")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .filter(|name| name.len() < 128)
            .map(str::to_owned)
            .collect();
        let zero = |key: &str| {
            entry
                .pointer(&format!("/pricing/{key}"))
                .and_then(|v| {
                    v.as_str()
                        .and_then(|v| v.parse::<f64>().ok())
                        .or_else(|| v.as_f64())
                })
                .is_some_and(|value| value == 0.0)
        };
        models.push(Model {
            id: id.into(),
            name: entry
                .get("display_name")
                .or_else(|| entry.get("name"))
                .and_then(Value::as_str)
                .filter(|name| name.len() < 2048)
                .unwrap_or(id)
                .into(),
            context_length: entry
                .get("context_length")
                .or_else(|| entry.get("max_input_tokens"))
                .and_then(Value::as_u64),
            max_output_tokens: entry
                .pointer("/top_provider/max_completion_tokens")
                .or_else(|| entry.get("max_output_tokens"))
                .and_then(Value::as_u64),
            supported_parameters,
            free: zero("prompt") && zero("completion"),
        });
    }
    models.sort_by(|a, b| {
        a.name
            .to_lowercase()
            .cmp(&b.name.to_lowercase())
            .then_with(|| a.id.cmp(&b.id))
    });
    models.dedup_by(|a, b| a.id == b.id);
    ensure!(
        !models.is_empty(),
        "Provider returned an empty model catalog"
    );
    Ok(models)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn body(options: GenerationOptions) -> Value {
        let mut body = json!({"model": "m"});
        options.apply(&mut body, false).unwrap();
        body
    }
    #[test]
    fn picked_effort_reaches_the_request() {
        let high = GenerationOptions { reasoning_effort: Some("high".into()), ..Default::default() };
        assert_eq!(body(high.clone())["reasoning_effort"], "high");
        assert_eq!(body(GenerationOptions { reasoning_object: true, ..high })["reasoning"], json!({"effort": "high"}));
        // Ultra is clamped to the strongest wire level, and Thinking off disables reasoning.
        assert_eq!(body(GenerationOptions { reasoning_effort: Some("ultra".into()), ..Default::default() })["reasoning_effort"], "max");
        assert_eq!(body(GenerationOptions { reasoning_effort: Some("none".into()), reasoning_object: true, ..Default::default() })["reasoning"], json!({"enabled": false}));
        assert!(body(GenerationOptions::default()).get("reasoning_effort").is_none());
    }
}
