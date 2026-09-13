"""Explicit lifecycle methods for the assignment-bound worker store.

The owner reserves identity before construction. Closing the local store remains
separate from ending its session and settling execution (including deferred work).
"""


class RuntimeSessionLifecycleMixin:
    def create_session(self, session_id, source, *, model=None, model_config=None,
                       system_prompt=None, user_id=None, session_key=None, chat_id=None,
                       chat_type=None, thread_id=None, parent_session_id=None, cwd=None,
                       profile_name=None, git_repo_root=None, origin_json=None, display_name=None):
        self._session(session_id)
        payload = dict(source=source, model=model, model_config=model_config, system_prompt=system_prompt,
                       user_id=user_id, session_key=session_key, chat_id=chat_id, chat_type=chat_type,
                       thread_id=thread_id, parent_session_id=parent_session_id, cwd=cwd,
                       profile_name=profile_name, git_repo_root=git_repo_root,
                       origin_json=origin_json, display_name=display_name)
        return self._apply('session.create', payload)['value']

    def set_session_title(self, session_id, title):
        self._session(session_id)
        return self._title_result(self._apply('session.title', {'title': title, 'source': 'user'}))

    def set_auto_title(self, session_id, title, *, source):
        if source not in ('derived', 'llm'):
            raise ValueError(f'invalid automatic title source: {source!r}')
        self._session(session_id)
        return self._title_result(self._apply('session.title', {'title': title, 'source': source}))

    @staticmethod
    def _title_result(result):
        if result.get('error'):
            raise ValueError(result['error'])
        return result['value']

    def get_session_title_source(self, session_id):
        row = self.get_session(session_id)
        return row.get('title_source') if row.get('title') is not None else None

    def set_session_title_source(self, session_id, source):
        if source not in ('user', 'derived', 'llm'):
            raise ValueError(f'invalid title source: {source!r}')
        self._session(session_id)
        return self._apply('session.title_source', {'source': source})['value']

    def get_next_title_in_lineage(self, base_title):
        return self._apply('session.next_title', {'base_title': base_title})['value']

    def touch_session_activity(self, session_id, ts=None, *, description=None, provenance=None):
        self._session(session_id)
        self._apply('session.activity', {'ts': ts, 'description': description, 'provenance': provenance})

    def clear_session_activity_labels(self, session_id):
        self._session(session_id)
        self._apply('session.activity_clear', {})

    def update_session_billing_route(self, session_id, *, provider, base_url, billing_mode=None):
        self._session(session_id)
        # Every preceding usage call already owns a receipt. _apply refuses a
        # pending or failed delta, so a route switch cannot overtake accounting.
        self._apply('session.billing_route', {'provider': provider, 'base_url': base_url,
                                              'billing_mode': billing_mode})

    def set_latest_user_api_content(self, session_id, content, api_content):
        self._session(session_id)
        return self._apply('session.api_content', {'content': content, 'api_content': api_content})['value']

    def end_session(self, session_id, end_reason):
        self._session(session_id)
        self._apply('session.end', {'end_reason': end_reason})

    def session_lifecycle_statuses(self, session_ids):
        ids = [sid for sid in (session_ids or []) if sid]
        for sid in ids:
            self._session(sid)
        if not ids:
            return {}
        return {self.scope['session_id']: self._apply('session.lifecycle', {})['value']}
