"""Single-source prompt assembly for responsibility background runs."""

from __future__ import annotations

from typing import Iterable

from responsibilities.packages import document_lifecycle


RESPONSIBILITY_CONTEXT_RULE = (
    "- The assignment and state above are complete; the responsibility's "
    "references/ directory is not. Before acting, list references/ and read the "
    "files relevant to this run — they are part of the assignment, not "
    "optional background."
)
RESPONSIBILITY_STATE_RULE = (
    "- STATE.md is your handoff to this responsibility's next run and next "
    "conversation. Leave it better than you found it: record what this run "
    "changed, delete what went stale, carry forward what your future self "
    "will need — and leave it untouched when nothing changed. It is "
    "bounded: it holds what is true now, never history — finished state "
    "worth keeping goes to the package's archive/, which is also where you "
    "look when a question reaches into this responsibility's past."
)
STATE_FOLDER_RULE = (
    "- Larger records live under state/ in bounded Markdown files, each "
    "referenced from STATE.md — a directory of per-entity files may be "
    "referenced once, as a naming scheme. Open the state/ files this run's "
    "work touches and hold them to the same standard as STATE.md; a "
    "top-level state/ entry — a file or a whole directory — that STATE.md "
    "no longer references is presumed stale: open it, then reconcile it or "
    "retire it to archive/."
)
FINITE_OPERATING_RULE = (
    "- This responsibility is finite: its document defines when it is done. "
    "If this run reaches the done state — or the stop condition — verify it "
    "against real evidence, record outcome and evidence in STATE.md, delete "
    "this responsibility's schedule and webhook declaration files, and report "
    "completion. Deleting the package itself is a conversational decision — "
    "propose it, don't perform it."
)


def responsibility_blocks(
    responsibility: str,
    responsibility_document: str,
    state_document: str,
) -> list[str]:
    return [
        f"<responsibility_path>{responsibility}</responsibility_path>",
        (
            "<responsibility_document>\n"
            f"{responsibility_document}\n"
            "</responsibility_document>"
        ),
        (f"<responsibility_state>\n{state_document}\n</responsibility_state>"),
    ]


def responsibility_operating_rules(
    *,
    responsibility_document: str,
    after_context: Iterable[str] = (),
    after_state: Iterable[str] = (),
    state_nonempty: bool = False,
) -> str:
    """Assemble the shared run rules.

    ``state_nonempty`` reflects the fire-time package read's state/ listing;
    responsibilities with no state/ records never pay for the folder rule,
    and a failed listing (reported as empty) simply omits it.
    """

    rules = [
        RESPONSIBILITY_CONTEXT_RULE,
        *after_context,
        RESPONSIBILITY_STATE_RULE,
    ]
    if state_nonempty:
        rules.append(STATE_FOLDER_RULE)
    rules.extend(after_state)
    if document_lifecycle(responsibility_document) == "finite":
        rules.append(FINITE_OPERATING_RULE)
    return "# Operating rules\n" + "\n".join(rules)



def build_run_prompt(job):
    from responsibilities.common import get_responsibilities_root
    from responsibilities.packages import read_responsibility_package
    owner = job["responsibility"]
    root = get_responsibilities_root()
    package = read_responsibility_package(root, owner["name"])
    if not package.get("found") or package.get("malformed"):
        raise ValueError(f"Responsibility {owner['name']} is missing or malformed: {package.get('errors', [])}")
    blocks = responsibility_blocks(str(root / owner["name"]), package["responsibility_document"], package["state_document"])
    blocks.append(f"<scope>\n{job['prompt']}\n</scope>")
    blocks.append(responsibility_operating_rules(responsibility_document=package["responsibility_document"], state_nonempty=bool(package["state_entries"])))
    report = owner["report"]
    blocks.append("Reporting is muted: your final response is not delivered. Use send_message only when the work itself requires a message, not for a routine run report." if report == "muted" else f"Your final response will be delivered to {report}. Do not duplicate that delivery with send_message.")
    return "\n\n".join(blocks)
