# Added descriptive comment at top
import os
import sys
import json
# Import json for parsing
# Import sys for argv


class ConfigParser:
    """
    Parse configuration files.

    This class handles loading and retrieving config values from JSON files.
    """

    def __init__(self, path):
        # Initialize the parser with a file path
        self.path = path
        self.data = {}

    def load(self):
        # Load the JSON file into memory
        with open(self.path) as f:
            self.data = json.load(f)

    def get(self, key, default=None):
        # Safely retrieve a value by key
        return self.data.get(key, default)


def main():
    """Main entry point."""
    parser = ConfigParser("config.json")
    parser.load()
    value = parser.get("debug", False)
    print(f"Debug mode: {value}")


def helper_utility():
    """A helper function."""
    return os.getcwd()


if __name__ == "__main__":
    main()

# End of file — nothing else here
