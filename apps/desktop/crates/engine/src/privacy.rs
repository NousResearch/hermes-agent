//! Model-side text filtering. Credentials used to authenticate the HTTPS
//! request and the user's unmodified local transcript stay outside this filter.
use regex::Regex;
use serde_json::Value;
use std::{borrow::Cow,sync::LazyLock};

const HIDDEN:&str="[REDACTED]";
static BLOCK:LazyLock<Regex>=LazyLock::new(||Regex::new(r"(?s)-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----").unwrap());
static TOKEN:LazyLock<Regex>=LazyLock::new(||Regex::new(r"\b(?:sk-(?:ant-|proj-|or-v1-)?[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{24,}|github_pat_[A-Za-z0-9_]{26,}|AKIA[A-Z0-9]{16}|xox[baprs]-[A-Za-z0-9-]{12,}|AIza[A-Za-z0-9_-]{30,})\b").unwrap());
static BEARER:LazyLock<Regex>=LazyLock::new(||Regex::new(r"(?i)(\b(?:Bearer|Basic)[ \t]+)[A-Za-z0-9._~+/=-]{8,}").unwrap());
static ASSIGN:LazyLock<Regex>=LazyLock::new(||Regex::new(r#"(?im)(\b[A-Za-z0-9_.-]*(?:api[_-]?key|access[_-]?token|secret|password|passwd|refresh[_-]?token|auth[_-]?token|private[_-]?key)["']?\s*[:=]\s*)(?:"[^"\r\n]+"|'[^'\r\n]+'|[^,;\s}\]]+)"#).unwrap());

pub(crate) fn contains_secret(text:&str)->bool{
    BLOCK.is_match(text)||TOKEN.is_match(text)||BEARER.is_match(text)||ASSIGN.is_match(text)
}
fn redact(text:&mut String){
    let block=BLOCK.replace_all(text,HIDDEN);
    let token=TOKEN.replace_all(&block,HIDDEN);
    let bearer=BEARER.replace_all(&token,"${1}[REDACTED]");
    let result=ASSIGN.replace_all(&bearer,"${1}[REDACTED]");
    if matches!(result,Cow::Owned(_))||matches!(bearer,Cow::Owned(_))||matches!(token,Cow::Owned(_))||matches!(block,Cow::Owned(_)){*text=result.to_string();}
}
fn sensitive_name(key:&str)->bool{
    let key=key.to_ascii_lowercase().replace(['-','.'],"_");
    ["api_key","apikey","password","passwd","client_secret","access_token","refresh_token","private_key","auth_token"].iter().any(|field|key==*field||key.ends_with(&format!("_{field}")))
}
fn redact_value(value:&mut Value){
    match value {
        Value::String(s)=>redact(s),
        Value::Array(a)=>a.iter_mut().for_each(redact_value),
        Value::Object(obj)=>for (key,v) in obj{
            if sensitive_name(key)&&!v.is_null(){*v=Value::String(HIDDEN.into());}else{redact_value(v);}
        },
        _=>{}
    }
}
pub(crate) fn redact_messages(messages:&mut [Value]){
    for message in messages {
        let tool=message["role"]=="tool";
        match &mut message["content"] {
            Value::String(s) if tool=>if let Ok(mut data)=serde_json::from_str::<Value>(s){
                redact_value(&mut data);*s=data.to_string();
            }else{redact(s)},
            Value::String(s)=>redact(s),
            Value::Array(parts)=>for p in parts {if let Some(Value::String(s))=p.get_mut("text"){redact(s)}},
            _=>{}
        }
        if let Some(calls)=message.get_mut("tool_calls").and_then(Value::as_array_mut){
            for call in calls {
                if let Some(Value::String(args))=call["function"].get_mut("arguments") && let Ok(mut data)=serde_json::from_str::<Value>(args){
                    redact_value(&mut data);*args=data.to_string();
                }
            }
        }
    }
}
