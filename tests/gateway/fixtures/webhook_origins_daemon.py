"""Real gateway with a concrete test consumer registered on its native plugin manager."""
from hermes_cli.plugins import get_plugin_manager
from gateway.run import main

manager = get_plugin_manager()
manager.discover_and_load()


def consume(**event):
    if event.get('platform') == 'webhook' and event.get('text') == 'HOOK_INPUT':
        print('HOOK_CONSUMED', event['message_id'], flush=True)
        return {'action': 'handled', 'reply': 'HOOK_ACTUAL_REPLY'}
    return None


manager._hooks.setdefault('post_gateway_admission', []).append(consume)
main()
