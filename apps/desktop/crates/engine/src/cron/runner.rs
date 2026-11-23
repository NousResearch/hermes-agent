use super::{Claim, Outcome, Store};
use crate::{CancellationToken, Config, Entry, Event, Session, Status};
use anyhow::{Result, ensure};
use async_channel::Sender;
use chrono::Utc;
use std::time::{Duration, Instant};

/// Execute one durable claim. The caller supplies the approval consumer: every
/// tool call still follows Session's approval protocol. No new process is used.
pub async fn execute(
    store: Store,
    claim: Claim,
    mut config: Config,
    events: Sender<Event>,
    cancel: CancellationToken,
) -> Result<Outcome> {
    let outcome = execute_inner(&store, &claim, &mut config, &events, &cancel)
        .await
        .unwrap_or_else(|error| Outcome::Failed {
            error: format!("{error:#}"),
        });
    store.finish(&claim, outcome.clone(), Utc::now())?;
    Ok(outcome)
}
async fn execute_inner(
    store: &Store,
    claim: &Claim,
    config: &mut Config,
    events: &Sender<Event>,
    cancel: &CancellationToken,
) -> Result<Outcome> {
    store.begin(claim, Utc::now())?;
    let job = &claim.job;
    ensure!(
        job.deliver == "local",
        "This native scheduler delivers to This desktop only"
    );
    for field in ["script", "monitor_script", "monitor_url"] {
        ensure!(
            !job.extra
                .get(field)
                .is_some_and(|value| value.as_str().is_some_and(|text| !text.trim().is_empty())),
            "This native scheduler does not execute scripts or monitors"
        );
    }
    ensure!(
        !job.extra
            .get("no_agent")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
        "Script-only execution is not available in this native scheduler"
    );
    if let Some(provider) = job
        .provider
        .as_deref()
        .filter(|provider| !provider.is_empty())
    {
        let base = config.base_url.trim_end_matches('/');
        let matches = provider.trim_end_matches('/') == base
            || match provider {
                "openai" => base == crate::OPENAI,
                "openrouter" => base == crate::OPENROUTER,
                "anthropic" => base == crate::ANTHROPIC,
                "nous" | "nous-portal" => base == crate::portal::INFERENCE.trim_end_matches('/'),
                _ => false,
            };
        ensure!(
            matches,
            "This job's pinned provider is not the configured provider. Select that provider before running this job"
        );
    }
    if let Some(base) = job
        .extra
        .get("base_url")
        .and_then(serde_json::Value::as_str)
        .filter(|base| !base.is_empty())
    {
        ensure!(
            base.trim_end_matches('/') == config.base_url.trim_end_matches('/'),
            "A job cannot reuse this provider's credentials at another endpoint"
        );
    }
    if let Some(model) = &job.model {
        config.model = model.clone();
    }
    config.validate()?;
    let default_workspace = store.profile_root().join("cron").join("workspace");
    let workspace = if let Some(workspace) = &job.workdir {
        workspace.clone()
    } else {
        std::fs::create_dir_all(&default_workspace)?;
        default_workspace
    };
    let mut session =
        Session::open_scheduled(&claim.run.session_dir, &workspace, store.profile_root())?;
    let (send, receive) = async_channel::unbounded();
    let mut inactivity = false;
    let turn_result = {
        let turn = session.turn(config.clone(), job.prompt.clone(), send, cancel.clone());
        tokio::pin!(turn);
        let mut last_activity = Instant::now();
        let mut heartbeat = tokio::time::interval(Duration::from_secs(2));
        loop {
            tokio::select! {
                result=&mut turn=>break result,
                event=receive.recv()=>if let Ok(event)=event{last_activity=Instant::now();if events.send(event).await.is_err(){cancel.cancel();}},
                _=heartbeat.tick()=>{
                    if store.cancellation_requested(&claim.run.id).unwrap_or(true){cancel.cancel();}
                    if last_activity.elapsed()>=Duration::from_secs(600){inactivity=true;cancel.cancel();}
                }
            }
        }
    };
    while let Ok(event) = receive.try_recv() {
        let _ = events.try_send(event);
    }
    turn_result?;
    let Some(turn) = session.history().turns.last() else {
        return Ok(Outcome::Failed {
            error: "The scheduled run did not create a conversation turn".into(),
        });
    };
    Ok(match &turn.status {
        Status::Complete => Outcome::Completed {
            output: turn
                .entries
                .iter()
                .filter_map(|entry| {
                    if let Entry::Assistant { text, .. } = entry {
                        Some(text.as_str())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join("\n"),
        },
        Status::Interrupted => Outcome::Cancelled {
            reason: if inactivity {
                "Stopped after ten minutes without activity"
            } else {
                "Stopped by the user"
            }
            .into(),
        },
        Status::Failed(reason) => Outcome::Failed {
            error: reason.clone(),
        },
        Status::Running => Outcome::Failed {
            error: "Run ended without a terminal status".into(),
        },
    })
}
