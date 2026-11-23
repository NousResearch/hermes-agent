use anyhow::{Context, Result, bail, ensure};
use hermes_engine::{Config, Session};
use hermes_gateway::{Gateway, GatewayConfig};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    net::IpAddr,
    path::{Path, PathBuf},
};

#[tokio::main]
async fn main() -> Result<()> {
    let mut port = 8765;
    let mut host: IpAddr = "127.0.0.1".parse()?;
    let mut root = None;
    let mut token_file = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--port" => {
                port = args
                    .next()
                    .context("--port requires a port number")?
                    .parse()
                    .context("Invalid port")?
            }
            "--host" => {
                host = args
                    .next()
                    .context("--host requires a loopback IP address")?
                    .parse()
                    .context("Invalid IP address")?
            }
            "--data-dir" => {
                root = Some(PathBuf::from(
                    args.next().context("--data-dir requires a path")?,
                ))
            }
            "--token-file" => {
                token_file = Some(PathBuf::from(
                    args.next().context("--token-file requires a path")?,
                ))
            }
            "--help" | "-h" => {
                println!(
                    "Hermes Rust loopback gateway\n\nUsage: hermes-gateway [--host 127.0.0.1] [--port 8765] [--data-dir PATH] [--token-file PATH]\n\nProvider: HERMES_RUST_BASE_URL, HERMES_RUST_MODEL, HERMES_RUST_API_KEY\nAuthentication: HERMES_RUST_GATEWAY_TOKEN or a private generated token file\nSupports a documented subset of Hermes' legacy and hermes-gateway-v1 WebSocket protocols.\nUse rpc.discover for the supported method list. Ctrl-C or SIGTERM shuts down and saves active turns."
                );
                return Ok(());
            }
            _ => bail!("Unknown option: {arg}. Use --help"),
        }
    }
    ensure!(
        host.is_loopback(),
        "Only loopback gateway listeners are supported"
    );
    let root = root.unwrap_or(Session::default_dir()?);
    fs::create_dir_all(&root).context("Could not create the data directory")?;
    let mut config = GatewayConfig::new(&root, Config::from_env());
    config.bind = (host, port).into();
    let token_path = if let Ok(token) = std::env::var("HERMES_RUST_GATEWAY_TOKEN") {
        ensure!(
            !token.is_empty(),
            "HERMES_RUST_GATEWAY_TOKEN must not be empty"
        );
        config.token = token;
        None
    } else {
        let file = token_file.unwrap_or_else(|| root.join("gateway.token"));
        config.token = read_or_create_token(&file)?;
        Some(file)
    };
    let gateway = Gateway::start(config).await?;
    println!(
        "{}",
        serde_json::json!({"implementation":"hermes-rust","http_url":gateway.http_url(),"websocket_url":gateway.websocket_url(),"token_file":token_path,"protocol":"hermes-gateway-v1","scope":"documented subset"})
    );
    #[cfg(unix)]
    {
        let mut terminate =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
        tokio::select! {_=tokio::signal::ctrl_c()=>{},_=terminate.recv()=>{}}
    }
    #[cfg(not(unix))]
    tokio::signal::ctrl_c().await?;
    gateway.shutdown().await
}

fn read_or_create_token(path: &Path) -> Result<String> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            ensure!(
                metadata.is_file() && !metadata.file_type().is_symlink(),
                "Token file must be a regular file"
            );
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                ensure!(
                    metadata.permissions().mode() & 0o077 == 0,
                    "Token file permissions must be private (chmod 600 {})",
                    path.display()
                );
            }
            let token = fs::read_to_string(path).context("Could not read gateway token")?;
            ensure!(
                !token.trim().is_empty(),
                "Gateway token file must not be empty"
            );
            Ok(token.trim().into())
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
                fs::create_dir_all(parent)?;
            }
            let token = format!(
                "{}{}",
                uuid::Uuid::new_v4().simple(),
                uuid::Uuid::new_v4().simple()
            );
            let mut options = OpenOptions::new();
            options.create_new(true).write(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            let mut file = options
                .open(path)
                .context("Could not create private gateway token file")?;
            writeln!(file, "{token}")?;
            file.sync_all()?;
            Ok(token)
        }
        Err(error) => Err(error).context("Could not inspect token file"),
    }
}
