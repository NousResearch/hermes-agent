#!/usr/bin/env python3
import argparse
import json


def classify(s):
    pulse_query = s.get('pulseQuery', False)
    effective_recent = s.get('recentState')
    minutes_since_assistant = s.get('minutesSinceAssistant')

    if pulse_query and effective_recent == 'pulse-check':
        effective_recent = s.get('prePulseRecentState') or 'idle'
    if pulse_query and minutes_since_assistant is not None and minutes_since_assistant <= 1:
        minutes_since_assistant = s.get('prePulseMinutesSinceAssistant', minutes_since_assistant)

    blocked = s.get('blocked') or effective_recent == 'blocked'
    if blocked:
        return 'blocked'
    if s.get('deliveryDue') and s.get('pendingActions', 0) >= 1:
        return 'busy'
    if s.get('hasStartedWork') and s.get('pendingActions', 0) >= 1:
        return 'busy'
    if (s.get('runningTask') and s.get('queuedMessages', 0) >= 1) or \
       (effective_recent == 'working' and (minutes_since_assistant or 999) <= 5) or \
       s.get('queuedMessages', 0) >= 3:
        return 'busy'
    if s.get('activeProject') and s.get('pendingActions', 0) >= 1:
        return 'light'
    if s.get('runningTask') or 1 <= s.get('queuedMessages', 0) <= 2 or (minutes_since_assistant or 999) <= 15:
        return 'light'
    if (not s.get('runningTask') and not s.get('blocked') and s.get('queuedMessages', 0) == 0 and (minutes_since_assistant or 0) > 15):
        return 'idle'
    return 'unknown'


def interruptibility(status, s):
    if status == 'blocked':
        return 'high' if s.get('waitingExternal') else 'medium'
    if status == 'busy' and s.get('deliveryDue'):
        return 'low'
    return {
        'idle': 'high',
        'light': 'medium',
        'busy': 'low',
        'unknown': 'medium',
    }.get(status, 'medium')


def accept_new(status, s):
    if status == 'busy' and s.get('deliveryDue'):
        return 'no'
    return {
        'idle': 'yes',
        'light': 'yes',
        'busy': 'caution',
        'blocked': 'caution',
        'unknown': 'caution',
    }.get(status, 'caution')


def build_signals(s, status):
    out = []
    if s.get('pulseQuery'):
        out.append('baseline-mode')
    if s.get('runningTask'):
        out.append('running-task')
    if s.get('activeProject'):
        out.append('active-project')
    if s.get('hasStartedWork'):
        out.append('started-work')
    if s.get('blocked'):
        out.append('blocked')
    if s.get('waitingExternal'):
        out.append('waiting-external')
    if s.get('deliveryDue'):
        out.append('delivery-due')
    if s.get('pendingActions', 0):
        out.append(f"pending:{s.get('pendingActions')}")
    if s.get('queuedMessages', 0):
        out.append(f"queued:{s.get('queuedMessages')}")
    if s.get('prePulseRecentState'):
        out.append(f"pre:{s.get('prePulseRecentState')}")
    elif s.get('recentState'):
        out.append(f"recent:{s.get('recentState')}")
    if not out:
        out.append('few-signals')
    return out[:8]


def main():
    p = argparse.ArgumentParser(description='Evaluate lightweight agent pulse')
    p.add_argument('--json', dest='json_text', required=True, help='Signal JSON object')
    args = p.parse_args()
    s = json.loads(args.json_text)
    status = classify(s)
    result = {
        'status': status,
        'interruptibility': interruptibility(status, s),
        'acceptNewTask': accept_new(status, s),
        'reason': f'rule-based:{status}',
        'signals': build_signals(s, status),
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
