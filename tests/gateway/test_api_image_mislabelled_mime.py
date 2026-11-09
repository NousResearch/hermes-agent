"""A data URL whose label disagrees with its image bytes (``data:image/png`` carrying JPEG, a
common SDK default) is admitted under the sniffed type, as main forwarded it to the provider."""
import base64
from pathlib import Path

from gateway.session_api_turn import admit_api_turn

JPEG = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9'


def test_mislabelled_image_commits_under_sniffed_type(api, owner):
    url = 'data:image/png;base64,' + base64.b64encode(JPEG).decode()
    parts = [{'type': 'text', 'text': 'look'}, {'type': 'image_url', 'image_url': {'url': url}}]
    _, _, row = admit_api_turn(api, session_id='api-mislabel', user_message=parts, conversation_history=[])
    media = row['payload']['api_turn_v1']['media']
    assert len(media) == 1 and Path(media[0]['path']).read_bytes() == JPEG
    assert Path(media[0]['path']).suffix == '.jpg'
