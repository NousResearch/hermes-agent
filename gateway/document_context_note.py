"""The context note prepended to a user turn when they attach a document.

The note is pure prompt text, so it belongs beside the inbound media plumbing rather than in the
``gateway.run`` facade (which is over its line budget). Two properties the note has to keep:

- It must not name tooling: it is built while the inbound message is prepared, *before* the turn's
  agent exists (and with no knowledge of the toolsets a cached agent is pinned to), so a deployment
  that disables the ``terminal`` toolset would be told to call a tool it does not have — and the
  ``ocr-and-documents`` skill the old wording named is not part of the bundled library at all.
- It must not call readable text a binary format: ``message/rfc822`` (``.eml``) and the text
  ``application/*`` documents adapters cache (``.json``/``.xml``/``.yaml``/``.toml``) are read, not
  extracted, so they take the text-document wording.
"""

_TEXT_READABLE_DOCUMENT_TYPES = frozenset({
    "message/rfc822", "application/json", "application/xml", "application/yaml", "application/toml",
})


def build_document_context_note(
    display_name: str, agent_path: str, mtype: str, *, content_inlined: bool = True,
) -> str:
    """Context note for one inbound attachment. ``content_inlined=False`` = cached without content,
    so tell the agent to read it. Binary docs must say *extract* the text; "ask the user" made it punt.
    Tools and skills stay unnamed here — see the module docstring for why."""
    text_readable = mtype.startswith("text/") or mtype in _TEXT_READABLE_DOCUMENT_TYPES
    if text_readable and content_inlined:
        return (
            f"[The user sent a text document: '{display_name}'. Its content has been included below. "
            f"The file is also saved at: {agent_path}]")
    if text_readable:
        return (
            f"[The user sent a text document: '{display_name}'. It is saved at: {agent_path}. "
            f"Its content is not inlined here. Read the cached file yourself before answering "
            f"when the user's request involves its contents.]")
    return (
        f"[The user sent a document: '{display_name}'. It is saved at: {agent_path}. "
        f"Its text is not inlined here (it's a binary format such as PDF or DOCX). "
        f"To read it, extract the document's text yourself with a document-reading tool "
        f"available to this run — before answering, instead of asking the user to "
        f"paste the contents.]")
