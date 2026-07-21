#!/usr/bin/env python3
import argparse
import json


def main():
    p = argparse.ArgumentParser(description='Render Agent Pulse fixed output card')
    p.add_argument('--json', required=True, help='Pulse result JSON')
    args = p.parse_args()
    data = json.loads(args.json)
    print('Agent Pulse')
    print(f"status: {data.get('status', 'unknown')}")
    print(f"interruptibility: {data.get('interruptibility', 'medium')}")
    print(f"acceptNewTask: {data.get('acceptNewTask', 'caution')}")
    print(f"reason: {data.get('reason', 'insufficient signals')}")


if __name__ == '__main__':
    main()
