"""Thread-root error classification and request timeouts in the Mattermost adapter."""

import asyncio

from plugins.platforms.mattermost.adapter import MattermostAdapter


def test_unrelated_invalid_post_400_is_not_a_broken_thread_root():
    body = '{"id":"api.post.create_post.props.app_error","message":"Invalid post properties"}'
    assert not MattermostAdapter._is_broken_thread_root_error(400, body)


def test_invalid_root_id_400_is_a_broken_thread_root():
    body = '{"id":"api.post.create_post.root_id.app_error","message":"Invalid RootId parameter."}'
    assert MattermostAdapter._is_broken_thread_root_error(400, body)


def test_request_timeout_returns_no_status_instead_of_raising():
    class _TimeoutSession:
        def post(self, *_args, **_kwargs):
            raise TimeoutError()

    adapter = MattermostAdapter.__new__(MattermostAdapter)
    adapter._base_url = "https://mm.example.com"
    adapter._session = _TimeoutSession()
    adapter._headers = dict

    data, status, _error = asyncio.run(adapter._api("POST", "posts", {"message": "hi"}))

    assert data == {}
    assert status == 0
