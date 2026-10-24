"""Document attachment descriptions supplied to the next agent turn."""


def _build_document_context_note(
    display_name: str,
    agent_path: str,
    mtype: str,
    *,
    content_inlined: bool = True,
    filename_requires_path_only: bool = False,
) -> str:
    """Context note prepended to a user turn when they attach a document.
    ``content_inlined=False`` = cached without content, so tell the agent to read it. Binary docs must
    say *extract* the text; "ask the user" made it punt."""
    if filename_requires_path_only:
        return (
            f"[The user sent a file attachment: '{display_name}'. It is saved at: {agent_path}. "
            f"Its content was not automatically inlined because the filename indicates it may "
            f"contain credentials or a private key. Read the saved file only if the user's "
            f"request requires it.]"
        )
    if mtype.startswith("text/") and content_inlined:
        return (
            f"[The user sent a text document: '{display_name}'. Its content has been included below. "
            f"The file is also saved at: {agent_path}]")
    if mtype.startswith("text/"):
        return (
            f"[The user sent a text document: '{display_name}'. It is saved at: {agent_path}. "
            f"Its content is not inlined here. Read the cached file yourself before answering "
            f"when the user's request involves its contents.]")
    return (
        f"[The user sent a document: '{display_name}'. It is saved at: {agent_path}. "
        f"Its text is not inlined here (it's a binary format such as PDF or DOCX). "
        f"To read it, extract the document's text yourself — for example with the "
        f"terminal tool or the ocr-and-documents skill — before answering, instead "
        f"of asking the user to paste the contents.]")
