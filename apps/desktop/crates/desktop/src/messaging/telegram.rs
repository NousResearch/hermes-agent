//! Native Telegram Bot API transport. No Python process or gateway wrapper.
//! Protocol: https://core.telegram.org/bots/api#getupdates and #sendmessage.
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::BTreeSet, fmt, future::Future, pin::Pin, sync::Arc, time::Duration};
use tokio_util::sync::CancellationToken;

#[derive(Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub(crate) struct Token(String);
impl Token {
    pub fn parse(value: &str) -> Result<Self, Error> {
        let value = value.trim();
        let valid = value.len() <= 256
            && value.split_once(':').is_some_and(|(id, secret)| {
                !id.is_empty()
                    && id.bytes().all(|byte| byte.is_ascii_digit())
                    && secret.len() >= 30
                    && secret
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
            });
        if !valid {
            return Err(Error::invalid(
                "Enter the complete bot token from @BotFather.",
            ));
        }
        Ok(Self(value.to_owned()))
    }
    fn value(&self) -> &str {
        &self.0
    }
    fn redact(&self, value: &str) -> String {
        value
            .replace(&self.0, "[redacted]")
            .chars()
            .take(300)
            .collect()
    }
}
impl fmt::Debug for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("TelegramToken([redacted])")
    }
}
impl<'de> Deserialize<'de> for Token {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = String::deserialize(deserializer)?;
        Self::parse(&value)
            .map_err(|_| serde::de::Error::custom("Invalid saved Telegram credential"))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ErrorKind {
    Invalid,
    Retryable,
    Cancelled,
    Rejected,
}
#[derive(Debug, Clone)]
pub(crate) struct Error {
    pub kind: ErrorKind,
    pub message: String,
    pub retry_after: Option<u64>,
}
impl Error {
    pub fn invalid(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Invalid,
            message: message.into(),
            retry_after: None,
        }
    }
    fn cancelled() -> Self {
        Self {
            kind: ErrorKind::Cancelled,
            message: "Telegram connection stopped".into(),
            retry_after: None,
        }
    }
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}
impl std::error::Error for Error {}

#[derive(Debug, Clone, Copy)]
pub(crate) enum Method {
    GetMe,
    GetWebhookInfo,
    GetUpdates,
    SendMessage,
}
impl Method {
    fn name(self) -> &'static str {
        match self {
            Self::GetMe => "getMe",
            Self::GetWebhookInfo => "getWebhookInfo",
            Self::GetUpdates => "getUpdates",
            Self::SendMessage => "sendMessage",
        }
    }
}
pub(crate) type RequestFuture<'a> = Pin<Box<dyn Future<Output = Result<Value, Error>> + Send + 'a>>;
/// The injectable boundary used by fake transports; tests need no HTTP server
/// and production has one fixed, verified HTTPS API origin.
pub(crate) trait Transport: Send + Sync {
    fn request<'a>(&'a self, token: &'a Token, method: Method, body: Value) -> RequestFuture<'a>;
}

struct HttpsTransport {
    client: reqwest::Client,
}
impl HttpsTransport {
    fn new() -> Result<Self, Error> {
        Ok(Self {
            client: reqwest::Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .connect_timeout(Duration::from_secs(10))
                .timeout(Duration::from_secs(35))
                .build()
                .map_err(|_| Error::invalid("Could not initialize the Telegram connection"))?,
        })
    }
}
impl Transport for HttpsTransport {
    fn request<'a>(&'a self, token: &'a Token, method: Method, body: Value) -> RequestFuture<'a> {
        Box::pin(async move {
            let response = self
                .client
                .post(format!(
                    "https://api.telegram.org/bot{}/{}",
                    token.value(),
                    method.name()
                ))
                .json(&body)
                .send()
                .await
                .map_err(|error| Error {
                    kind: ErrorKind::Retryable,
                    message: if error.is_timeout() {
                        "Telegram request timed out"
                    } else {
                        "Could not reach Telegram"
                    }
                    .into(),
                    retry_after: Some(3),
                })?;
            let status = response.status();
            let mut response = response;
            let mut bytes = Vec::new();
            while let Some(chunk) = response.chunk().await.map_err(|_| Error {
                kind: ErrorKind::Retryable,
                message: "Could not read Telegram's reply".into(),
                retry_after: Some(3),
            })? {
                if bytes.len() + chunk.len() > 2 * 1024 * 1024 {
                    return Err(Error::invalid(
                        "Telegram reply exceeds the native safe-load limit",
                    ));
                }
                bytes.extend_from_slice(&chunk);
            }
            let envelope: Value = serde_json::from_slice(&bytes)
                .map_err(|_| Error::invalid("Telegram returned an unreadable reply"))?;
            if status.is_success() && envelope.get("ok") == Some(&Value::Bool(true)) {
                return envelope
                    .get("result")
                    .cloned()
                    .ok_or_else(|| Error::invalid("Telegram reply has no result"));
            }
            let retryable = status.as_u16() == 429 || status.is_server_error();
            let description = envelope
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or("Telegram rejected the request");
            Err(Error {
                kind: if retryable {
                    ErrorKind::Retryable
                } else {
                    ErrorKind::Rejected
                },
                message: token.redact(description),
                retry_after: envelope
                    .pointer("/parameters/retry_after")
                    .and_then(Value::as_u64)
                    .map(|seconds| seconds.clamp(1, 60)),
            })
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Identity {
    pub id: i64,
    pub username: String,
    pub name: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Incoming {
    pub bot_id: i64,
    pub update_id: i64,
    pub message_id: i64,
    pub user_id: i64,
    pub chat_id: i64,
    pub sender: String,
    pub text: String,
    pub timestamp: i64,
}
#[derive(Debug, Clone)]
pub(crate) struct PollBatch {
    pub messages: Vec<Incoming>,
    pub next_offset: i64,
    pub ignored: usize,
}
#[derive(Debug, Clone)]
pub(crate) struct Receipt {
    pub message_id: i64,
    pub timestamp: i64,
}

#[derive(Clone)]
pub(crate) struct Client {
    transport: Arc<dyn Transport>,
    token: Token,
    allowed: BTreeSet<i64>,
}
impl fmt::Debug for Client {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("TelegramClient([redacted])")
    }
}
impl Client {
    pub fn credential(&self) -> &Token {
        &self.token
    }
    pub fn new(token: Token, allowed: BTreeSet<i64>) -> Result<Self, Error> {
        Self::with_transport(token, allowed, Arc::new(HttpsTransport::new()?))
    }
    pub fn with_transport(
        token: Token,
        allowed: BTreeSet<i64>,
        transport: Arc<dyn Transport>,
    ) -> Result<Self, Error> {
        if allowed.is_empty()
            || allowed
                .iter()
                .any(|id| *id <= 0 || *id > 4_503_599_627_370_495)
        {
            return Err(Error::invalid(
                "Add at least one numeric Telegram user ID to the allowlist.",
            ));
        }
        Ok(Self {
            transport,
            token,
            allowed,
        })
    }
    async fn request(
        &self,
        method: Method,
        body: Value,
        cancel: &CancellationToken,
    ) -> Result<Value, Error> {
        tokio::select! {biased;_=cancel.cancelled()=>Err(Error::cancelled()),result=self.transport.request(&self.token,method,body)=>result.map_err(|mut error|{error.message=self.token.redact(&error.message);error})}
    }
    pub async fn verify(&self, cancel: &CancellationToken) -> Result<Identity, Error> {
        let user = self.request(Method::GetMe, json!({}), cancel).await?;
        if user.get("is_bot") != Some(&Value::Bool(true)) {
            return Err(Error::invalid("This credential is not a Telegram bot"));
        }
        let id = user
            .get("id")
            .and_then(Value::as_i64)
            .filter(|id| *id > 0)
            .ok_or_else(|| Error::invalid("Telegram did not identify this bot"))?;
        let username = user
            .get("username")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        let name = user
            .get("first_name")
            .and_then(Value::as_str)
            .unwrap_or("Telegram")
            .to_owned();
        let webhook = self
            .request(Method::GetWebhookInfo, json!({}), cancel)
            .await?;
        if webhook
            .get("url")
            .and_then(Value::as_str)
            .is_some_and(|url| !url.is_empty())
        {
            return Err(Error::invalid(
                "This bot has an existing webhook. Remove it in that integration or use a separate bot before connecting here.",
            ));
        }
        Ok(Identity { id, username, name })
    }
    pub async fn poll(
        &self,
        bot_id: i64,
        offset: i64,
        cancel: &CancellationToken,
    ) -> Result<PollBatch, Error> {
        let value=self.request(Method::GetUpdates,json!({"offset":offset.max(0),"limit":100,"timeout":20,"allowed_updates":["message"]}),cancel).await?;
        let updates = value
            .as_array()
            .ok_or_else(|| Error::invalid("Telegram returned an invalid update batch"))?;
        let mut batch = PollBatch {
            messages: vec![],
            next_offset: offset,
            ignored: 0,
        };
        for update in updates {
            let Some(update_id) = update
                .get("update_id")
                .and_then(Value::as_i64)
                .filter(|id| *id >= 0)
            else {
                batch.ignored += 1;
                continue;
            };
            batch.next_offset = batch.next_offset.max(update_id.saturating_add(1));
            if update_id < offset {
                continue;
            }
            let Some(message) = update.get("message") else {
                batch.ignored += 1;
                continue;
            };
            let user_id = message.pointer("/from/id").and_then(Value::as_i64);
            let chat_id = message.pointer("/chat/id").and_then(Value::as_i64);
            let private = message.pointer("/chat/type").and_then(Value::as_str) == Some("private");
            if !private
                || user_id != chat_id
                || user_id.is_none_or(|id| !self.allowed.contains(&id))
                || message.pointer("/from/is_bot") == Some(&Value::Bool(true))
            {
                batch.ignored += 1;
                continue;
            }
            let Some(text) = message.get("text").and_then(Value::as_str).filter(|text| {
                !text.trim().is_empty() && text.len() <= 32768 && text.chars().count() <= 4096
            }) else {
                batch.ignored += 1;
                continue;
            };
            let Some(message_id) = message.get("message_id").and_then(Value::as_i64) else {
                batch.ignored += 1;
                continue;
            };
            let sender = message
                .pointer("/from/username")
                .and_then(Value::as_str)
                .or_else(|| message.pointer("/from/first_name").and_then(Value::as_str))
                .map(str::to_owned)
                .unwrap_or_else(|| user_id.unwrap().to_string());
            batch.messages.push(Incoming {
                bot_id,
                update_id,
                message_id,
                user_id: user_id.unwrap(),
                chat_id: chat_id.unwrap(),
                sender,
                text: text.to_owned(),
                timestamp: message.get("date").and_then(Value::as_i64).unwrap_or(0),
            });
        }
        Ok(batch)
    }
    pub async fn send(
        &self,
        user_id: i64,
        text: &str,
        cancel: &CancellationToken,
    ) -> Result<Receipt, Error> {
        if !self.allowed.contains(&user_id) {
            return Err(Error::invalid("This Telegram user is not on the allowlist"));
        }
        if text.trim().is_empty() || text.chars().count() > 4096 {
            return Err(Error::invalid("Enter a reply of 1–4096 characters"));
        }
        let result = self
            .request(
                Method::SendMessage,
                json!({"chat_id":user_id,"text":text,"link_preview_options":{"is_disabled":true}}),
                cancel,
            )
            .await?;
        let message_id = result
            .get("message_id")
            .and_then(Value::as_i64)
            .ok_or_else(|| Error::invalid("Telegram did not confirm delivery"))?;
        Ok(Receipt {
            message_id,
            timestamp: result.get("date").and_then(Value::as_i64).unwrap_or(0),
        })
    }
}

pub(crate) fn parse_allowlist(values: &[String]) -> Result<BTreeSet<i64>, Error> {
    let mut ids = BTreeSet::new();
    for value in values.iter().flat_map(|value| value.split([',', '\n'])) {
        let value = value.trim();
        if value.is_empty() {
            continue;
        }
        let id = value
            .parse::<i64>()
            .ok()
            .filter(|id| *id > 0 && *id <= 4_503_599_627_370_495)
            .ok_or_else(|| {
                Error::invalid("Allowed Telegram users must be numeric user IDs, one per box.")
            })?;
        ids.insert(id);
    }
    if ids.is_empty() {
        return Err(Error::invalid(
            "Add at least one allowed Telegram user ID before connecting.",
        ));
    }
    if ids.len() > 1000 {
        return Err(Error::invalid(
            "The native Telegram allowlist supports up to 1000 users",
        ));
    }
    Ok(ids)
}
