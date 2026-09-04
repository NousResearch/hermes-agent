mod app;
mod complete;
mod composer_vim;
mod config;
mod events;
mod fs_tree;
mod layout;
mod optimistic;
mod palette;
mod paste;
mod platform;
mod rpc;
mod shell;
mod skill_md;
mod slash;
mod state;
mod terminal;
mod tips;
mod ui;

use anyhow::{Context, Result};
use clap::Parser;
use std::sync::Arc;
use tokio::sync::{Mutex, Notify};

use config::{init_tracing, validate_python_gateway, Cli, LaunchConfig};
use rpc::GatewayClient;
use state::AppState;
use terminal::{new_terminal, TerminalGuard};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let cfg = LaunchConfig::from_cli(cli)?;
    init_tracing(&cfg.hermes_home)?;
    validate_python_gateway(&cfg.python, &cfg.src_root)?;

    tracing::info!(
        python = %cfg.python.display(),
        src_root = %cfg.src_root.display(),
        hermes_home = %cfg.hermes_home.display(),
        "starting hermes-tui"
    );

    let client = GatewayClient::spawn(&cfg)
        .await
        .context("spawning tui_gateway")?;
    let session_id = if let Some(id) = cfg.resume.clone() {
        id
    } else {
        client
            .create_session(&cfg.title, &cfg.cwd)
            .await
            .context("session.create")?
    };

    let state = Arc::new(Mutex::new(AppState::new()));
    {
        let mut s = state.lock().await;
        s.session_id = Some(session_id.clone());
        s.startup_resume = cfg.resume.clone();
        s.session_title = cfg.title.clone();
        s.session_started = std::time::Instant::now();
        s.metrics.cwd = cfg.cwd.display().to_string();
        s.hermes_home = cfg.hermes_home.clone();
        let pal = crate::ui::theme::load_saved(&s.hermes_home);
        crate::ui::theme::apply(pal);
        s.theme_id = pal.id.to_string();
        s.tips_open = crate::tips::load_open(&s.hermes_home);
        s.tip_index = crate::tips::start_index();
        s.tip_shown_at = std::time::Instant::now();
    }

    let mut event_rx = client.subscribe_events();
    let state_events = state.clone();
    let redraw = Arc::new(Notify::new());
    let redraw_events = redraw.clone();
    tokio::spawn(async move {
        use tokio::sync::broadcast::error::{RecvError, TryRecvError};
        loop {
            let first = match event_rx.recv().await {
                Ok(evt) => evt,
                Err(RecvError::Lagged(n)) => {
                    let mut s = state_events.lock().await;
                    s.note_lagged(n);
                    redraw_events.notify_one();
                    continue;
                }
                Err(RecvError::Closed) => break,
            };
            let mut s = state_events.lock().await;
            crate::events::apply_event(&mut s, &first);
            loop {
                match event_rx.try_recv() {
                    Ok(evt) => crate::events::apply_event(&mut s, &evt),
                    Err(TryRecvError::Empty) | Err(TryRecvError::Closed) => break,
                    Err(TryRecvError::Lagged(n)) => {
                        s.note_lagged(n);
                        break;
                    }
                }
            }
            if s.dirty {
                redraw_events.notify_one();
            }
        }
    });

    let client = Arc::new(client);

    let run_res = {
        let _guard = TerminalGuard::enter()?;
        let mut terminal = new_terminal()?;
        let out = app::run(&mut terminal, state, client.clone(), redraw).await;
        terminal::flush_terminal(&mut terminal);
        out
    };

    let _ = client.close_session(&session_id).await;
    client.shutdown().await;

    run_res?;
    println!("Hermes TUI exited.");
    Ok(())
}
