"""从官方辅助调用观察器投影 MoA 状态，不持有模型结果或修改模型请求。"""


def observe_auxiliary(bound, payload, *, completed):
    task = payload.get("aux_task")
    request_id = payload.get("api_request_id")
    if bound is None or task not in {"moa_reference", "moa_aggregator"} or not request_id:
        return
    adapter, turn = bound
    session_id = payload.get("session_id")
    with turn.lock:
        if (
            turn.final
            or turn.cancelled
            or (session_id and turn.agent_session_id and session_id != turn.agent_session_id)
        ):
            return
        request_id = str(request_id)
        attempt = int(payload.get("retry_count") or 0)
        previous = turn.moa_requests.get(request_id)
        if (
            not completed
            and previous is None
            and task == "moa_reference"
            and any(r["task"] == "moa_aggregator" for r in turn.moa_requests.values())
        ):
            # MoA 在 pre_api_request 之前准备参考意见；下一波以真实参考请求为界。
            turn.moa_requests.clear()
        if previous and (
            attempt < previous["attempt"]
            or (attempt == previous["attempt"] and previous["completed"])
        ):
            return
        # post 仅更新本轮实际开始过的请求，拒绝上一轮迟到的结果。
        if completed and previous is None:
            return
        label = "/".join(str(payload.get(k) or "") for k in ("provider", "model")).strip("/")
        turn.moa_requests[request_id] = {
            "task": task,
            "label": label or "unknown",
            "attempt": attempt,
            "completed": completed,
            "status": "failed"
            if completed and payload.get("error")
            else ("done" if completed and not payload.get("streaming") else "running"),
        }
        refs = [r for r in turn.moa_requests.values() if r["task"] == "moa_reference"]
        aggregators = [r for r in turn.moa_requests.values() if r["task"] == "moa_aggregator"]
        degraded = any(r["status"] == "failed" for r in refs)
        phase = "references"
        if aggregators:
            phase = "degraded_aggregating" if degraded else "aggregating"
        turn.orchestration = {
            "kind": "moa",
            "phase": phase,
            "references": [{"label": r["label"], "status": r["status"]} for r in refs],
            "refs_done": sum(r["status"] != "running" for r in refs),
            "refs_total": len(refs),
            "aggregator": aggregators[-1]["label"] if aggregators else "",
        }
    adapter.schedule_refresh(turn)
