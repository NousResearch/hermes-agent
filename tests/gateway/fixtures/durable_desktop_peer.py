"""JSON-lines test peer for the real durable Desktop RPC adapter. No listener or real credentials."""
import json
import os
from pathlib import Path
import sys
import uuid
from dataclasses import asdict

wire = sys.stdout  # tui_gateway may redirect its own stdout when imported.
from pytest import MonkeyPatch
from hermes_cli import _early_recovery
from tests.fixtures.gateway_owner import GatewayOwner
from gateway.session_authority import SessionAuthority

home = Path(os.environ['HERMES_HOME'])
assert home.is_dir() and str(home) != str(Path.home() / '.hermes')
patch = MonkeyPatch()
patch.setattr(_early_recovery, 'restore_interrupted_pull', lambda root: False)
patch.setattr(SessionAuthority, '_schedule_admitted', lambda *args, **kwargs: None)
owner = GatewayOwner(home, patch)
owner.authority.instance_id = uuid.uuid4().hex
life = owner._run(owner._open())
try:
    for line in sys.stdin:
        request = json.loads(line)
        try:
            method = request['method']
            if method == 'fixture.restart':
                owner.close()
                owner = GatewayOwner(home, patch)
                owner.authority.instance_id = uuid.uuid4().hex
                life = owner._run(owner._open())
                result = {'result': True}
            elif method == 'fixture.reconnect':
                owner._run(life.connection.close())
                life = owner._run(owner._open())
                result = {'result': True}
            elif method == 'fixture.claim':
                claim = owner.persistence.admissions.claim(epoch=owner.authority.epoch,
                                                           session_id=request['params']['session_id'])
                result = {'result': asdict(claim) if claim else None}
            elif method == 'fixture.principal':
                from gateway.session_controls import AuthorityConnection
                owner._run(life.connection.close())
                async def change_principal():
                    return AuthorityConnection(owner.authority, object(), {'user_id': request['params']['user']})
                life.connection = owner._run(change_principal())
                result = {'result': True}
            elif method == 'fixture.ledger':
                rows = owner.persistence.admissions.match(request_id=request['params']['request_id'])
                result = {'result': [asdict(row) for row in rows]}
            else:
                result = owner._run(life.connection.dispatch(request))
            wire.write(json.dumps({'id': request['id'], **result}) + '\n')
            wire.flush()
        except Exception as error:
            wire.write(json.dumps({'id': request['id'], 'error': {'message': str(error)}}) + '\n')
            wire.flush()
finally:
    owner.close()
    patch.undo()
