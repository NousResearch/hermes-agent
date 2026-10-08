"""Copy context identities only when copying the exact prompt they describe."""


def inherited_context_manifest(conn, parent_session_id, prompt_hash, prompt, manifest=None):
    if manifest is not None or not parent_session_id or not prompt:
        return manifest
    row = conn.execute(
        """SELECT context_file_identities FROM sessions WHERE id = ? AND
           (system_prompt_hash = ? OR (system_prompt_hash IS NULL AND system_prompt = ?))""",
        (parent_session_id, prompt_hash, prompt),
    ).fetchone()
    return row[0] if row is not None else None
