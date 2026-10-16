"""The paused gateways' tree gate for the update's checkout-writing steps (Windows pause, #132338).

``_checkout_move`` binds one git step (switch, merge, reset, upstream sync, rollback, stash, churn
cleanup) to the pause record BEFORE git writes a file; ``update_cmd`` re-exports these names.
"""

from contextlib import contextmanager

from hermes_cli import update_pause_record as _pause_record


@contextmanager
def _checkout_move(token, *targets, paths=(), revert=False):
    """Run one checkout-writing git step (switch, merge, reset, upstream sync, rollback, stash) with
    the paused gateways' tree gate bound to it BEFORE git writes a file: the commit(s) it can move
    to and the tracked *paths* it rewrites in place (stash push/apply, ``reset --hard``,
    ``checkout -- <paths>``; *revert*: every tracked change it finds, put back to HEAD), on a
    baseline taken at the HEAD it starts from. Without that, a step
    git leaves half-written at an unmoved HEAD is judged against whatever refs a later fetch left,
    or against no baseline at all once an earlier step moved HEAD. A failed record write raises:
    the step must not run.

    The first step from the pause's own HEAD extends the pause's baseline (it knows the edits that
    were already there). A later step gets a baseline of its own, retired once git returns with the
    tree whole: HEAD moved and every file the move wrote holds the new commit, the tree exactly as
    the step found it, or the step reported ``whole`` on the dict this yields (a stash restore that
    applied cleanly changes the tree by design). A move is judged the moment git returns (see
    ``_settle_checkout_move``)."""
    step: dict = {}
    targets = sorted({t for t in targets if t})
    pause_id = (token or {}).get("pause_id")
    root = _pause_record.install_root() if pause_id and (targets or paths or revert) else None
    found = _pause_record.tree_state(root) if root else None
    if found is None:  # nothing paused, nothing bound, or not a git checkout: no tree gate to bind
        yield step
        return
    paths = {*paths, *(found["dirty_at_pause"] or [] if revert else ())}
    baselines = token.setdefault("baselines", [])
    move_id = f"{pause_id}@{found['pre_sha']}"
    baseline = next((b for b in baselines if b.get("pre_sha") == found["pre_sha"]
                     and b.get("pause_id") in (pause_id, move_id)), None)
    if baseline is None:
        baseline = {"pause_id": move_id, **found}
        baselines.append(baseline)
    for key, bound in (("move_targets", targets), ("move_paths", paths)):
        if bound:
            baseline[key] = sorted({*baseline.get(key, []), *bound})
    _pause_record.write({**token, "resume_needed": True})
    try:
        yield step
    finally:
        _settle_checkout_move(token, baseline, found, root, step)


def _moves_for(token):
    """``_checkout_move`` bound to *token*: the ``checkout_move`` the git and stash helpers take."""
    return lambda *targets, **bound: _checkout_move(token, *targets, **bound)


def _settle_checkout_move(token: dict, baseline: dict, found: dict, root, step: dict) -> None:
    """Record what the returned step left. A moved HEAD gets its verdict now: git can move HEAD
    past a file it failed to write, and only now are the move's paths untouched by later steps
    (dependency syncs, stash restores). A step's own baseline is retired when the tree is whole."""
    now = _pause_record.tree_state(root)
    moved = now["pre_sha"] != found["pre_sha"]
    torn = _pause_record.torn_by_move(root, found, now["dirty_at_pause"]) if moved else None
    if moved and torn is not None:  # unknown stays unjudged: the gate judges it at resume time
        baseline.setdefault("landed", {})[now["pre_sha"]] = torn
    whole = torn == [] if moved else step.get("whole") or all(now[key] == found[key] for key in found)
    if baseline["pause_id"] != token["pause_id"] and whole:
        token["baselines"] = [b for b in token["baselines"] if b is not baseline]
    _pause_record.write({**token, "resume_needed": True})
