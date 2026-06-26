from __future__ import annotations

from .models import ContextResult, Observation


def _bullet_list(title: str, items: list[str]) -> list[str]:
    if not items:
        return []
    return [f"{title}:", *[f"- {item}" for item in items]]


def _observation_line(observation: Observation) -> str:
    title = observation.title.strip()
    if observation.files:
        file_list = ", ".join(file.file_path for file in observation.files[:3])
        return f"[{observation.observation_type}] {title} (files: {file_list})"
    return f"[{observation.observation_type}] {title}"


def merge_context_results(primary: ContextResult, fallback: ContextResult, *, limit: int) -> ContextResult:
    """Merge targeted and fallback retrieval results, preserving primary order."""
    merged: list[Observation] = []
    seen: set[tuple[object, str, str]] = set()
    for source in (primary.observations, fallback.observations):
        for obs in source:
            key = (obs.id, obs.session_id, obs.title)
            if key in seen:
                continue
            seen.add(key)
            merged.append(obs)
            if len(merged) >= limit:
                break
        if len(merged) >= limit:
            break

    changed_files: list[str] = []
    for path in (*primary.changed_files, *fallback.changed_files):
        if path not in changed_files:
            changed_files.append(path)

    follow_ups: list[str] = []
    for item in (*primary.suggested_follow_ups, *fallback.suggested_follow_ups):
        if item not in follow_ups:
            follow_ups.append(item)

    observations = tuple(merged)
    return ContextResult(
        observations=observations,
        decisions=tuple(obs for obs in observations if obs.observation_type == "decision"),
        changed_files=tuple(changed_files),
        session_fact=primary.session_fact or fallback.session_fact,
        suggested_follow_ups=tuple(follow_ups),
    )


def format_context_result(result: ContextResult, *, max_observations: int = 4) -> str:
    """Format sidecar retrieval into a compact prompt-ready text block."""
    lines: list[str] = ["Memory sidecar context:"]

    fact = result.session_fact
    if fact and fact.user_goal:
        lines.append(f"User goal: {fact.user_goal}")
    if fact and fact.latest_summary:
        lines.append(f"Latest summary: {fact.latest_summary}")

    decisions = [obs.title.strip() for obs in result.decisions[:max_observations]]
    lines.extend(_bullet_list("Recent decisions", decisions))

    next_steps = [
        obs.title.strip()
        for obs in result.observations
        if obs.observation_type == "next_step"
    ][:max_observations]
    lines.extend(_bullet_list("Next steps", next_steps))

    relevant_observations = [
        _observation_line(obs)
        for obs in result.observations
        if obs.observation_type not in {"decision", "next_step"}
    ][:max_observations]
    lines.extend(_bullet_list("Relevant observations", relevant_observations))

    if result.changed_files:
        lines.append(f"Changed files: {', '.join(result.changed_files[:8])}")
    if result.suggested_follow_ups:
        lines.extend(_bullet_list("Useful follow-ups", list(result.suggested_follow_ups[:3])))

    if len(lines) == 1:
        return ""
    return "\n".join(lines)
