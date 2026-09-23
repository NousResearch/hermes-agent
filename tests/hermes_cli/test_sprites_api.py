import json
from unittest.mock import Mock

import pytest

from hermes_cli import sprites_api


@pytest.mark.parametrize('failure', ['error', 'exit'])
def test_failed_service_event_is_not_masked_by_completion(monkeypatch, failure):
    response = Mock(status=200)
    response.getheader.return_value = 'application/x-ndjson'
    response.read.return_value = (json.dumps({'type': failure, 'exit_code': 1}) + '\n' + json.dumps({'type': 'complete'})).encode()
    connection = Mock()
    connection.getresponse.return_value = response
    monkeypatch.setattr(sprites_api, '_Connection', lambda *args, **kwargs: connection)
    with pytest.raises(RuntimeError, match='not acknowledged'):
        sprites_api.request('POST', '/services/gateway-default/start')
    connection.close.assert_called_once()
