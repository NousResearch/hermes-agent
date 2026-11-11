"""Physical barriers around real producers, never a fabricated admission or receipt."""
import os
from pathlib import Path
import signal
import threading

from gateway.run import GatewayRunner, main
from gateway import webhook_final_intent
from hermes_cli.plugins import get_plugin_manager

original_handoff = webhook_final_intent._handoff


def handoff(authority, value, *args):
    if value['origin_kind'] == 'completion_final' and value['message'].endswith('GROUP_FINAL_2'):
        Path(os.environ['TEST_BARRIER']).write_text(value['delivery_id'])
        os.kill(os.getpid(), signal.SIGSTOP)
    return original_handoff(authority, value, *args)


webhook_final_intent._handoff = handoff
original_arm = GatewayRunner.arm_process_watcher

# A shortened grace period exercises native pin withdrawal without a five-minute
# wall-clock wait. No receipt or admission bytes are supplied by the fixture.
from gateway import session_bot
if os.environ.get('TEST_RECEIPT_STAGE'):
    session_bot._PIN_GRACE_SECONDS = 0
original_receipt = session_bot.write_receipt


async def receipt(home, record, **kwargs):
    stage = os.environ.get('TEST_RECEIPT_STAGE')
    marker = Path(os.environ['TEST_BARRIER'])
    if stage == 'admitted' and record.get('admission_id') and not marker.exists():
        marker.write_text(record['delivery_id'])
        os.kill(os.getpid(), signal.SIGSTOP)
    answer = await original_receipt(home, record, **kwargs)
    if stage == 'pin' and not record.get('admission_id') and not marker.exists():
        marker.write_text(record['delivery_id'])
        os.kill(os.getpid(), signal.SIGSTOP)
    return answer


session_bot.write_receipt = receipt


def arm(self, watcher):
    # Exercise the existing raw-terminal watcher mode, rather than an agent-notify
    # completion. The terminal still creates its actual OS process and descriptor.
    if os.environ.get('TEST_RAW_WATCHER') == '1':
        watcher = dict(watcher, notify_on_complete=False, check_interval=0.1)
    return original_arm(self, watcher)


GatewayRunner.arm_process_watcher = arm
manager = get_plugin_manager()
manager.discover_and_load()


hook_release = threading.Event()


def consume(**event):
    if event.get('platform') == 'webhook' and event.get('text') == 'HOOK_BLOCK':
        path = Path(os.environ['TEST_HOOK_EFFECT'])
        with path.open('a') as output:
            output.write('hook side effect\n')
        # Kill the actual producer while its concrete consumer owns preparation.
        hook_release.wait(timeout=90)
        return {'action': 'handled', 'reply': 'NEVER_COMMITTED_HOOK_REPLY'}
    return None


manager._hooks.setdefault('post_gateway_admission', []).append(consume)
main()
