import os
import sys
import json
from pathlib import Path
from collections import OrderedDict


class BaseHandler:
    """Base handler class."""

    def __init__(self):
        self.handlers = []

    def register(self, handler):
        self.handlers.append(handler)


class DataProcessor(BaseHandler):
    """Process data files."""

    def __init__(self, source):
        super().__init__()
        self.source = source
        self.results = OrderedDict()

    def process(self):
        with open(self.source) as f:
            data = json.load(f)
        for key, value in data.items():
            self.results[key] = value.upper() if isinstance(value, str) else value
        return self.results

    def get_result(self, key):
        return self.results.get(key)


def main():
    processor = DataProcessor("data.json")
    result = processor.process()
    print(json.dumps(result, indent=2))


def validate_input(data):
    """Validate incoming data."""
    if not data:
        raise ValueError("Empty input")
    return True


def format_output(data):
    """Format output for display."""
    return json.dumps(data, indent=2)


async def fetch_remote(url):
    """Fetch data from remote URL."""
    return None


if __name__ == "__main__":
    main()
