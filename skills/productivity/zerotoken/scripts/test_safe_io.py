"""Minimal regression check for safe_io encoding fallbacks."""

import os
import tempfile

from safe_io import safe_read


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        gb_path = os.path.join(directory, "gb.txt")
        bom_path = os.path.join(directory, "bom.txt")
        with open(gb_path, "wb") as file:
            file.write("中文".encode("gb18030"))
        with open(bom_path, "wb") as file:
            file.write(b"\xef\xbb\xbf" + "中文".encode("utf-8"))
        assert safe_read(gb_path) == "中文"
        assert safe_read(bom_path) == "中文"


if __name__ == "__main__":
    main()
