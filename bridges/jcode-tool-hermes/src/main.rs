use std::env;
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug)]
struct Config {
    service_command: String,
    cwd: Option<String>,
    request_json: Option<String>,
    tool: Option<String>,
    args_json: Option<String>,
    id: Option<String>,
    session_id: Option<String>,
    task_id: Option<String>,
    allow_tools: Vec<String>,
    confirm_outbound_human_contact: bool,
    confirm_sensitive_person_data: bool,
    safety_override_reason: Option<String>,
}

fn usage() -> &'static str {
    "Usage: jcode-tool-hermes [options]\n\
\n\
Options:\n\
  --service-command <cmd>  Command to run Hermes service stdio wrapper\n\
  --cwd <dir>              Working directory for the service command\n\
  --request-json <json>    Complete hermes-service.v1 request object\n\
  --tool <name>            Hermes tool name when building a request\n\
  --args-json <json>       Hermes tool args object when building a request\n\
  --id <id>                Optional request id\n\
  --session-id <id>        Optional session id\n\
  --task-id <id>           Optional task id\n\
  --allow-tool <name>      Forwarded to service wrapper; repeatable\n\
  --confirm-outbound-human-contact\n\
  --confirm-sensitive-person-data\n\
  --safety-override-reason <reason>\n\
  --help"
}

fn take_value(args: &[String], index: &mut usize, flag: &str) -> Result<String, String> {
    *index += 1;
    args.get(*index)
        .cloned()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_args() -> Result<Config, String> {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut config = Config {
        service_command: "python3 scripts/hermes_service_bridge.py stdio".to_string(),
        cwd: None,
        request_json: None,
        tool: None,
        args_json: None,
        id: None,
        session_id: None,
        task_id: None,
        allow_tools: Vec::new(),
        confirm_outbound_human_contact: false,
        confirm_sensitive_person_data: false,
        safety_override_reason: None,
    };

    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--help" | "-h" => {
                println!("{}", usage());
                std::process::exit(0);
            }
            "--service-command" => {
                config.service_command = take_value(&args, &mut index, "--service-command")?;
            }
            "--cwd" => config.cwd = Some(take_value(&args, &mut index, "--cwd")?),
            "--request-json" => {
                config.request_json = Some(take_value(&args, &mut index, "--request-json")?);
            }
            "--tool" => config.tool = Some(take_value(&args, &mut index, "--tool")?),
            "--args-json" => config.args_json = Some(take_value(&args, &mut index, "--args-json")?),
            "--id" => config.id = Some(take_value(&args, &mut index, "--id")?),
            "--session-id" => {
                config.session_id = Some(take_value(&args, &mut index, "--session-id")?)
            }
            "--task-id" => config.task_id = Some(take_value(&args, &mut index, "--task-id")?),
            "--allow-tool" => {
                config
                    .allow_tools
                    .push(take_value(&args, &mut index, "--allow-tool")?);
            }
            "--confirm-outbound-human-contact" => {
                config.confirm_outbound_human_contact = true;
            }
            "--confirm-sensitive-person-data" => {
                config.confirm_sensitive_person_data = true;
            }
            "--safety-override-reason" => {
                config.safety_override_reason =
                    Some(take_value(&args, &mut index, "--safety-override-reason")?);
            }
            other => return Err(format!("unknown argument: {other}\n\n{}", usage())),
        }
        index += 1;
    }

    Ok(config)
}

fn json_escape(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len() + 8);
    for ch in raw.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn generated_id() -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0);
    format!("jcode-hermes-{millis}")
}

fn build_request(config: &Config) -> Result<String, String> {
    if let Some(request) = &config.request_json {
        let trimmed = request.trim();
        if !trimmed.starts_with('{') {
            return Err("--request-json must be a JSON object".to_string());
        }
        return Ok(trimmed.to_string());
    }

    let tool = config
        .tool
        .as_deref()
        .ok_or_else(|| "--tool is required unless --request-json is provided".to_string())?;
    let args_json = config.args_json.as_deref().unwrap_or("{}").trim();
    if !args_json.starts_with('{') {
        return Err("--args-json must be a JSON object".to_string());
    }

    let id = config.id.clone().unwrap_or_else(generated_id);
    let mut fields = vec![
        "\"type\":\"hermes_service_request\"".to_string(),
        format!("\"id\":\"{}\"", json_escape(&id)),
        format!("\"tool\":\"{}\"", json_escape(tool)),
        format!("\"args\":{args_json}"),
    ];
    if let Some(session_id) = &config.session_id {
        fields.push(format!("\"session_id\":\"{}\"", json_escape(session_id)));
    }
    if let Some(task_id) = &config.task_id {
        fields.push(format!("\"task_id\":\"{}\"", json_escape(task_id)));
    }
    if config.confirm_outbound_human_contact {
        fields.push("\"confirm_outbound_human_contact\":true".to_string());
    }
    if config.confirm_sensitive_person_data {
        fields.push("\"confirm_sensitive_person_data\":true".to_string());
    }
    if let Some(reason) = &config.safety_override_reason {
        fields.push(format!(
            "\"safety_override_reason\":\"{}\"",
            json_escape(reason)
        ));
    }
    Ok(format!("{{{}}}", fields.join(",")))
}

fn shell_command(command: &str) -> Command {
    let mut cmd = Command::new("sh");
    cmd.arg("-c").arg(command);
    cmd
}

fn main() {
    let config = match parse_args() {
        Ok(config) => config,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(64);
        }
    };

    let request = match build_request(&config) {
        Ok(request) => request,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(64);
        }
    };

    let mut service_command = config.service_command.clone();
    for tool in &config.allow_tools {
        service_command.push_str(" --allow-tool ");
        service_command.push_str(&shell_quote(tool));
    }

    let mut command = shell_command(&service_command);
    if let Some(cwd) = &config.cwd {
        command.current_dir(cwd);
    }

    let mut child = match command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
    {
        Ok(child) => child,
        Err(error) => {
            eprintln!("failed to start Hermes service: {error}");
            std::process::exit(69);
        }
    };

    {
        let stdin = child.stdin.as_mut().expect("service stdin must be piped");
        if let Err(error) = writeln!(stdin, "{request}") {
            eprintln!("failed to write Hermes service request: {error}");
            let _ = child.kill();
            std::process::exit(74);
        }
    }
    drop(child.stdin.take());

    let stdout = child.stdout.take().expect("service stdout must be piped");
    let mut reader = BufReader::new(stdout);
    let mut response = String::new();
    if let Err(error) = reader.read_line(&mut response) {
        eprintln!("failed to read Hermes service response: {error}");
        let _ = child.kill();
        std::process::exit(74);
    }

    let status = match child.wait() {
        Ok(status) => status,
        Err(error) => {
            eprintln!("failed to wait for Hermes service: {error}");
            std::process::exit(74);
        }
    };

    print!("{response}");
    if response.contains("\"ok\": false") || response.contains("\"ok\":false") {
        std::process::exit(2);
    }
    if !status.success() {
        std::process::exit(status.code().unwrap_or(1));
    }
}

fn shell_quote(value: &str) -> String {
    if value
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | '/' | ':'))
    {
        return value.to_string();
    }
    let escaped = value.replace('\'', "'\\''");
    format!("'{escaped}'")
}
