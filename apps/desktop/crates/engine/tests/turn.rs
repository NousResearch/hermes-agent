use axum::{Json, Router, extract::State, response::{Sse, sse::Event as SseEvent}, routing::post};
use futures_util::stream;
use hermes_engine::{CancellationToken, Config, Entry, Event, Session, Status};
use serde_json::{Value, json};
use std::{convert::Infallible, sync::{Arc, Mutex}};
use tempfile::tempdir;

type Requests = Arc<Mutex<Vec<Value>>>;

async fn fake_model(State(requests): State<Requests>, Json(body): Json<Value>)
    -> Sse<impl futures_util::Stream<Item = Result<SseEvent, Infallible>>> {
    let mut requests = requests.lock().unwrap();
    let prior = body["messages"].as_array().unwrap().last().unwrap();
    let mut chunks = Vec::new();
    if prior["role"] == "tool" {
        assert_eq!(prior["tool_call_id"], "clock-1");
        let payload: Value = serde_json::from_str(prior["content"].as_str().unwrap()).unwrap();
        let output = payload.get("utc").or(payload.get("error")).unwrap().as_str().unwrap();
        for text in ["Clock result: ", output] {
            chunks.push(json!({"choices":[{"index":0,"delta":{"content":text},"finish_reason":null}]}));
        }
        chunks.push(json!({"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}));
    } else {
        let tools = body["tools"].as_array().unwrap();
        assert!(tools.iter().any(|tool| tool["function"]["name"] == "get_current_time"));
        chunks.push(json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"clock-1","type":"function","function":{"name":"get_current_","arguments":"{"}}]},"finish_reason":null}]}));
        chunks.push(json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"time","arguments":"}"}}]},"finish_reason":"tool_calls"}]}));
    }
    requests.push(body);
    let events = chunks.into_iter().map(|data| Ok(SseEvent::default().data(data.to_string())))
        .chain(std::iter::once(Ok(SseEvent::default().data("[DONE]"))));
    Sse::new(stream::iter(events))
}

async fn setup() -> (Config, Requests) {
    let seen = Requests::default();
    let server = Router::new().route("/v1/chat/completions", post(fake_model)).with_state(seen.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base_url = format!("http://{}/v1", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, server).await.unwrap(); });
    (Config { base_url, api_key: String::new(), model:"turn-test".into() }, seen)
}

#[tokio::test]
async fn approval_streams_real_clock_returns_result_and_reopens() {
    let (config, seen) = setup().await;
    let dir = tempdir().unwrap();
    manual(dir.path());
    let mut session = Session::open(dir.path()).unwrap();
    assert!(Session::open(dir.path()).is_err());
    let (tx, rx) = async_channel::unbounded();
    let task = tokio::spawn(async move {
        session.turn(config, "Read the UTC clock.".into(), tx, CancellationToken::new()).await.unwrap();
        session
    });
    let mut streamed = String::new();
    let mut pieces = 0;
    while let Ok(event) = rx.recv().await {
        match event {
            Event::Approval { call, answer } => {
                assert_eq!(call.name, "get_current_time");
                assert_eq!(seen.lock().unwrap().len(), 1);
                answer.send(true).await.unwrap();
            }
            Event::Text(chunk) => { pieces += 1; streamed.push_str(&chunk); }
            _ => {}
        }
    }
    let session = task.await.unwrap();
    let saved = session.history().clone();
    let turn = &saved.turns[0];
    assert_eq!(turn.status, Status::Complete);
    assert_eq!(seen.lock().unwrap().len(), 2);
    assert_eq!(pieces, 2);
    let Entry::Tool { approved:true, result, .. } = &turn.entries[1] else { panic!("expected approved tool result") };
    let payload: Value = serde_json::from_str(result).unwrap();
    let utc = payload["utc"].as_str().unwrap();
    assert!(chrono::DateTime::parse_from_rfc3339(utc).is_ok());
    assert_eq!(streamed, format!("Clock result: {utc}"));
    drop(session);
    let restored = Session::open(dir.path()).unwrap();
    assert_eq!(restored.history(), &saved);
}

#[tokio::test]
async fn denied_tool_sends_denial_and_never_returns_time() {
    let (config, _) = setup().await;
    let dir = tempdir().unwrap();
    manual(dir.path());
    let mut session = Session::open(dir.path()).unwrap();
    let (tx, rx) = async_channel::unbounded();
    let task = tokio::spawn(async move {
        session.turn(config, "Time?".into(), tx, CancellationToken::new()).await.unwrap();
        session
    });
    while let Ok(event) = rx.recv().await {
        if let Event::Approval { answer, .. } = event { answer.send(false).await.unwrap(); }
    }
    let session = task.await.unwrap();
    let turn = &session.history().turns[0];
    assert_eq!(turn.status, Status::Complete);
    assert!(matches!(&turn.entries[1], Entry::Tool { approved:false, result, .. } if result.contains("denied") && !result.contains("\"utc\"")));
}

#[tokio::test]
async fn cancelling_at_approval_never_executes_or_resends_tool() {
    let (config, seen) = setup().await;
    let dir = tempdir().unwrap();
    manual(dir.path());
    let mut session = Session::open(dir.path()).unwrap();
    let (tx, rx) = async_channel::unbounded();
    let cancel = CancellationToken::new();
    let stop = cancel.clone();
    let task = tokio::spawn(async move {
        session.turn(config, "Time?".into(), tx, cancel).await.unwrap();
    });
    while let Ok(event) = rx.recv().await {
        if let Event::Approval { answer, .. } = event {
            stop.cancel();
            answer.close();
        }
    }
    task.await.unwrap();
    let restored = Session::open(dir.path()).unwrap();
    let turn = &restored.history().turns[0];
    assert_eq!(turn.status, Status::Interrupted);
    assert!(!turn.entries.iter().any(|e| matches!(e,Entry::Tool{..})));
    assert_eq!(seen.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn profile_memory_write_is_reviewed_and_context_is_frozen_until_reopen() {
    async fn memory_model(State(seen): State<Requests>, Json(body): Json<Value>) -> Sse<impl futures_util::Stream<Item=Result<SseEvent,Infallible>>> {
        let mut requests=seen.lock().unwrap();
        let i=requests.len();
        assert!(body["tools"].as_array().unwrap().iter().any(|t|t["function"]["name"]=="memory"));
        requests.push(body);
        let delta=if i==0 {
            json!({"tool_calls":[{"index":0,"id":"mem-1","type":"function","function":{"name":"memory","arguments":r#"{"target":"user","action":"add","content":"Synthetic fixture prefers blue."}"#}}]})
        } else { json!({"content":"ok"}) };
        let chunks = [json!({"choices":[{"index":0,"delta":delta,"finish_reason":null}]}),json!({"choices":[{"index":0,"delta":{},"finish_reason":if i==0 {"tool_calls"} else {"stop"}}]})];
        Sse::new(stream::iter(chunks.into_iter().map(|j|Ok(SseEvent::default().data(j.to_string()))).chain(std::iter::once(Ok(SseEvent::default().data("[DONE]"))))))
    }
    let requests=Requests::default();
    let router=Router::new().route("/v1/chat/completions",post(memory_model)).with_state(requests.clone());
    let listener=tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let config=Config{base_url:format!("http://{}/v1",listener.local_addr().unwrap()), model:"fixture".into(),api_key:String::new()};
    tokio::spawn(async move { axum::serve(listener,router).await.unwrap(); });
    let profile=tempdir().unwrap(); let first=tempdir().unwrap(); let second=tempdir().unwrap();
    let mut session=Session::open_for_profile(first.path(),None,profile.path()).unwrap();
    let (tx,rx)=async_channel::unbounded(); let model=config.clone();
    let task=tokio::spawn(async move{session.turn(model,"Save my preference".into(),tx,CancellationToken::new()).await.unwrap(); session});
    let mut approved=false;
    while let Ok(event)=rx.recv().await {
        if let Event::Approval {call,answer}=event {
            assert_eq!(call.name,"memory");
            let review:Value=serde_json::from_str(&call.arguments).unwrap();
            assert_eq!(review["target"],"user");
            assert!(review["operations"].to_string().contains("Synthetic fixture prefers blue."));
            assert!(!profile.path().join("memories/USER.md").exists());
            approved=true; answer.send(true).await.unwrap();
        }
    }
    assert!(approved);
    let session=task.await.unwrap();
    assert_eq!(session.history().turns[0].status,Status::Complete);
    let mut reopened=Session::open_for_profile(second.path(),None,profile.path()).unwrap();
    let (tx,_rx)=async_channel::unbounded();
    reopened.turn(config,"Recall my preference".into(),tx,CancellationToken::new()).await.unwrap();
    let req=requests.lock().unwrap();
    assert_eq!(req.len(),3);
    assert!(!req[1]["messages"][0]["content"].as_str().unwrap().contains("Synthetic fixture prefers blue."));
    assert!(req[2]["messages"][0]["content"].as_str().unwrap().contains("Synthetic fixture prefers blue."));
}

/// These tests exercise the approval prompt, so they pin the Manual mode.
fn manual(dir: &std::path::Path) {
    let mut changes = serde_json::Map::new();
    changes.insert("approval_mode".into(), serde_json::json!("manual"));
    hermes_engine::settings::ProfileSettings::update(dir, &changes).unwrap();
}

/// A group round as the desktop runs it: one transcript, each member's turn in
/// its own profile. The second bot sees the first's reply and its own persona.
#[tokio::test]
async fn group_round_runs_each_member_in_its_profile_over_one_transcript() {
    async fn echo(State(requests): State<Requests>, Json(body): Json<Value>) -> Sse<impl futures_util::Stream<Item = Result<SseEvent, Infallible>>> {
        let system = body["messages"][0]["content"].as_str().unwrap().to_owned();
        let who = if system.contains("You are Ada") { "Ada" } else if system.contains("You are Grace") { "Grace" } else { "nobody" };
        requests.lock().unwrap().push(body);
        let chunks = [json!({"choices":[{"index":0,"delta":{"content":format!("{who} here")},"finish_reason":null}]}), json!({"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]})];
        Sse::new(stream::iter(chunks.into_iter().map(|d| Ok(SseEvent::default().data(d.to_string()))).chain(std::iter::once(Ok(SseEvent::default().data("[DONE]"))))))
    }
    let seen = Requests::default();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base_url = format!("http://{}/v1", listener.local_addr().unwrap());
    let app = Router::new().route("/v1/chat/completions", post(echo)).with_state(seen.clone());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap(); });
    let config = Config { base_url, api_key: String::new(), model: "group".into() };
    let root = tempdir().unwrap();
    let chat = root.path().join("chat");
    std::fs::create_dir_all(&chat).unwrap();
    for (bot, soul) in [("ada", "You are Ada, terse."), ("grace", "You are Grace, kind.")] {
        let profile = root.path().join("profiles").join(bot);
        std::fs::create_dir_all(&profile).unwrap();
        std::fs::write(profile.join("SOUL.md"), soul).unwrap();
    }
    for (bot, prompt) in [("ada", "Review this plan"), ("grace", "@Grace — your turn")] {
        let mut session = Session::open_for_profile(&chat, None, &root.path().join("profiles").join(bot)).unwrap();
        let (tx, rx) = async_channel::unbounded();
        let config = config.clone();
        let run = tokio::spawn(async move { session.turn(config, prompt.into(), tx, CancellationToken::new()).await.unwrap(); session });
        while rx.recv().await.is_ok() {}
        let session = run.await.unwrap();
        assert_eq!(session.history().turns.last().unwrap().status, Status::Complete);
        drop(session);
    }
    let history = Session::open(&chat).unwrap().history().clone();
    let replies: Vec<_> = history.turns.iter().map(|t| match &t.entries[0] { Entry::Assistant { text, .. } => text.clone(), _ => String::new() }).collect();
    assert_eq!(replies, ["Ada here", "Grace here"], "each member answered in its own persona");
    let second = &seen.lock().unwrap()[1];
    assert!(second["messages"].as_array().unwrap().iter().any(|m| m["content"] == "Ada here"), "Grace sees Ada's reply");
}

/// delegate_task: the parent fans out two children that run in their own saved
/// sessions without the delegation tool; their answers come back as the result.
#[tokio::test]
async fn delegation_runs_children_and_returns_their_answers() {
    async fn model(State(requests): State<Requests>, Json(body): Json<Value>) -> Sse<impl futures_util::Stream<Item = Result<SseEvent, Infallible>>> {
        let messages = body["messages"].as_array().unwrap().clone();
        let names: Vec<String> = body["tools"].as_array().unwrap().iter().map(|t| t["function"]["name"].as_str().unwrap().to_owned()).collect();
        let first_user = messages.iter().find(|m| m["role"] == "user").unwrap()["content"].as_str().unwrap_or_default().to_owned();
        let last = messages.last().unwrap().clone();
        requests.lock().unwrap().push(json!({"child": first_user.starts_with("You are a subagent"), "tools": names}));
        let chunks = if first_user.starts_with("You are a subagent") {
            let goal = first_user.split("Goal: ").nth(1).unwrap().lines().next().unwrap().to_owned();
            vec![json!({"choices":[{"index":0,"delta":{"content":format!("answer for {goal}")},"finish_reason":"stop"}]})]
        } else if last["role"] == "tool" {
            let result: Value = serde_json::from_str(last["content"].as_str().unwrap()).unwrap();
            let summaries: Vec<String> = result["results"].as_array().unwrap().iter().map(|r| r["summary"].as_str().unwrap().to_owned()).collect();
            vec![json!({"choices":[{"index":0,"delta":{"content":summaries.join(" | ")},"finish_reason":"stop"}]})]
        } else {
            let args = json!({"tasks":[{"goal":"alpha"},{"goal":"beta","context":"use beta notes"}]}).to_string();
            vec![json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"d1","type":"function","function":{"name":"delegate_task","arguments":args}}]},"finish_reason":"tool_calls"}]})]
        };
        Sse::new(stream::iter(chunks.into_iter().map(|d| Ok(SseEvent::default().data(d.to_string()))).chain(std::iter::once(Ok(SseEvent::default().data("[DONE]"))))))
    }
    let seen = Requests::default();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base_url = format!("http://{}/v1", listener.local_addr().unwrap());
    let app = Router::new().route("/v1/chat/completions", post(model)).with_state(seen.clone());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap(); });
    let config = Config { base_url, api_key: String::new(), model: "m".into() };
    let dir = tempdir().unwrap();
    let mut session = Session::open(dir.path()).unwrap();
    let (tx, rx) = async_channel::unbounded();
    let run = tokio::spawn(async move { session.turn(config, "Split this work".into(), tx, CancellationToken::new()).await.unwrap(); session });
    let mut finished = std::collections::BTreeMap::new();
    while let Ok(event) = rx.recv().await {
        if let Event::Subagent(p) = event { if p.status == "completed" { finished.insert(p.goal.clone(), p.summary.clone().unwrap()); } }
    }
    let session = run.await.unwrap();
    assert_eq!(finished.get("alpha").map(String::as_str), Some("answer for alpha"));
    assert_eq!(finished.get("beta").map(String::as_str), Some("answer for beta"));
    let turn = session.history().turns.last().unwrap();
    assert_eq!(turn.status, Status::Complete);
    let Entry::Assistant { text, .. } = turn.entries.last().unwrap() else { panic!() };
    assert_eq!(text, "answer for alpha | answer for beta");
    let requests = seen.lock().unwrap();
    assert!(requests.iter().filter(|r| r["child"] == true).all(|r| !r["tools"].as_array().unwrap().iter().any(|t| t == "delegate_task")), "children cannot delegate");
    assert!(requests.iter().any(|r| r["child"] == false && r["tools"].as_array().unwrap().iter().any(|t| t == "delegate_task")));
    assert_eq!(std::fs::read_dir(dir.path().join("subagents")).unwrap().count(), 2, "each child keeps its own session");
}
