use anyhow::{Context, Result, anyhow, bail};
use async_trait::async_trait;
use jcode_tool_core::{Tool, ToolContext};
use jcode_tool_types::ToolOutput;
use serde_json::{Value, json};
use std::process::Stdio;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;
use tokio::time::timeout;

#[derive(Clone, Debug)]
pub struct HermesToolConfig {
    pub service_command: Vec<String>,
    pub timeout_ms: u64,
}

impl HermesToolConfig {
    pub fn local_service() -> Self {
        Self {
            service_command: vec![
                "python3".to_string(),
                "scripts/hermes_service_bridge.py".to_string(),
                "stdio".to_string(),
            ],
            timeout_ms: 60_000,
        }
    }
}

#[derive(Clone, Debug)]
pub struct HermesNativeTool {
    name: String,
    hermes_tool: String,
    description: String,
    parameters_schema: Value,
    config: HermesToolConfig,
}

impl HermesNativeTool {
    pub fn new(
        name: impl Into<String>,
        hermes_tool: impl Into<String>,
        description: impl Into<String>,
        parameters_schema: Value,
        config: HermesToolConfig,
    ) -> Self {
        Self {
            name: name.into(),
            hermes_tool: hermes_tool.into(),
            description: description.into(),
            parameters_schema,
            config,
        }
    }

    pub fn web_search(config: HermesToolConfig) -> Self {
        Self::new(
            "hermes_web_search",
            "web_search",
            "Search the web through Hermes' provider-rich research stack.",
            json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                    "confirm_sensitive_person_data": {"type": "boolean"},
                    "safety_override_reason": {"type": "string"}
                },
                "required": ["query"]
            }),
            config,
        )
    }

    pub fn web_extract(config: HermesToolConfig) -> Self {
        Self::new(
            "hermes_web_extract",
            "web_extract",
            "Extract page content through Hermes' configured web providers.",
            json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "url": {"type": "string"},
                    "urls": {"type": "array", "items": {"type": "string"}},
                    "confirm_sensitive_person_data": {"type": "boolean"},
                    "safety_override_reason": {"type": "string"}
                }
            }),
            config,
        )
    }

    async fn call_hermes_service(&self, args: Value, ctx: &ToolContext) -> Result<Value> {
        let (program, command_args) = self
            .config
            .service_command
            .split_first()
            .ok_or_else(|| anyhow!("Hermes service command is empty"))?;

        let mut child = Command::new(program)
            .args(command_args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("failed to start Hermes service: {program}"))?;

        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("Hermes service stdin was unavailable"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("Hermes service stdout was unavailable"))?;

        let request = json!({
            "type": "hermes_service_request",
            "id": ctx.tool_call_id,
            "tool": self.hermes_tool,
            "args": args,
            "session_id": ctx.session_id,
            "task_id": ctx.message_id,
        });
        stdin.write_all(request.to_string().as_bytes()).await?;
        stdin.write_all(b"\n").await?;
        drop(stdin);

        let mut line = String::new();
        let read = timeout(
            Duration::from_millis(self.config.timeout_ms),
            BufReader::new(stdout).read_line(&mut line),
        )
        .await
        .context("Hermes service timed out")??;
        if read == 0 {
            bail!("Hermes service closed stdout without a response");
        }

        let response: Value = serde_json::from_str(&line)
            .with_context(|| format!("invalid Hermes service response: {line}"))?;
        let status = child.wait().await?;
        if !status.success() && response.get("ok") != Some(&Value::Bool(true)) {
            bail!("Hermes service exited with status {status}");
        }
        Ok(response)
    }
}

#[async_trait]
impl Tool for HermesNativeTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters_schema(&self) -> Value {
        self.parameters_schema.clone()
    }

    async fn execute(&self, input: Value, ctx: ToolContext) -> Result<ToolOutput> {
        let response = self.call_hermes_service(input, &ctx).await?;
        if response.get("ok") != Some(&Value::Bool(true)) {
            let message = response
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("Hermes service returned an error");
            bail!("{message}");
        }

        let result = response.get("result").cloned().unwrap_or(Value::Null);
        let output = match result {
            Value::String(text) => text,
            other => serde_json::to_string_pretty(&other)?,
        };

        Ok(ToolOutput::new(output)
            .with_title(format!("hermes:{}", self.hermes_tool))
            .with_metadata(response))
    }
}
