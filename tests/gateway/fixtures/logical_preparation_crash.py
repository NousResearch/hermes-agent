"""A real owner stopped abruptly after one durable preparation batch."""
import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace

from gateway.session_authority import initialize_session_authority
import gateway.session_logical_preparation as preparation
from hermes_state import SessionDB


async def main(path, ready):
    # Hold at the normal yield between batches so the parent can kill a known
    # committed checkpoint instead of racing a machine-speed sleep.
    preparation._BATCH_PAUSE = 30
    with SessionDB(path) as db:
        runner = SimpleNamespace(_draining=False, adapters={}, session_store=SimpleNamespace(),
                                 config=SimpleNamespace(multiplex_profiles=False))
        authority = await initialize_session_authority(runner, profile_id=str(path.parent),
                                                       instance_id='crashed-owner', db=db)
        while True:
            with db._read_ctx() as conn:
                row = conn.execute('SELECT live_cursor FROM logical_attempt_coverage').fetchone()
            if row is not None and row[0] == 128:
                ready.write_text('committed')
                await asyncio.Event().wait()
            if authority._logical_preparation_task.done():
                raise RuntimeError('preparation ended before crash checkpoint')
            await asyncio.sleep(.01)


asyncio.run(main(Path(sys.argv[1]), Path(sys.argv[2])))
