//! Opt-in live verification of the same messages the native controls dispatch.
use super::*;
use hermes_engine::Status;
use iced::futures::StreamExt;

#[tokio::test]
#[ignore = "uses Nous Portal OAuth and qwen/qwen4-max:free; requires a new temporary data directory"]
async fn nous_qwen_send_approve_stream_and_reopen() {
    let data = std::path::PathBuf::from(std::env::var_os("HERMES_RUST_DATA_DIR").expect("Set an empty temporary data directory"));
    assert!(data.is_absolute() && data.is_dir() && data.read_dir().unwrap().next().is_none());
    let (mut app, _) = Hermes::new();
    app.config.base_url = portal::INFERENCE.into();
    app.config.model = "qwen/qwen4-max:free".into();
    app.config.api_key.clear();
    assert!(app.ready(), "Nous Portal sign-in is required");
    app.composer = text_editor::Content::with_text("Call get_current_time exactly once with an empty arguments object. Then report the exact UTC timestamp you received.");
    let send = app.update(Message::Send);
    assert!(app.running, "Send must start the active turn");
    let mut actions = iced_runtime::task::into_stream(send).unwrap();
    let mut granted = 0;
    let mut chunks = 0;
    while let Some(action) = actions.next().await {
        if let iced_runtime::Action::Output(message) = action {
            if matches!(message, Message::Engine(_, Event::Text(_))) { chunks += 1; }
            let _ = app.update(message);
            if let Some(approval) = &app.approval {
                assert_eq!(approval.call.name, "get_current_time");
                assert_eq!(serde_json::from_str::<serde_json::Value>(&approval.call.arguments).unwrap(),serde_json::json!({}));
                granted += 1;
                let _ = app.update(Message::Approve(true));
            }
        }
    }
    assert!(!app.running);
    let turn = app.history.turns.last().unwrap();
    assert_eq!(turn.status, Status::Complete, "Turn ended with {:?}",turn.status);
    assert_eq!(granted,1);
    assert!(chunks > 0);
    assert_eq!(turn.model,"qwen/qwen4-max:free");
    let utc = turn.entries.iter().find_map(|e| match e {
        Entry::Tool { approved:true,name,result,.. } if name=="get_current_time" => {
            serde_json::from_str::<serde_json::Value>(result).ok()?["utc"].as_str().map(str::to_owned)
        }
        _=>None,
    }).expect("Persisted approved clock result");
    assert!(turn.entries.iter().any(|e| matches!(e, Entry::Assistant{text,..} if text.contains(&utc))));
    let expected=app.history.clone();
    drop(actions);
    drop(app);
    let (reopened, _) = Hermes::new();
    assert_eq!(reopened.history,expected);
    println!("Desktop send, approval, {chunks} streamed text events, and reopened UTC result {utc} verified.");
}
