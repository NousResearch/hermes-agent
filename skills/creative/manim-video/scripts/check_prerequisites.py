"""Read-only, cross-platform prerequisite check for Manim video production."""
import argparse
import importlib.util
import shutil


def missing_requirements(no_latex=False):
    missing = []
    if importlib.util.find_spec("manim") is None:
        missing.append("manim")
    commands = ["ffmpeg"] if no_latex else ["ffmpeg", "latex", "dvisvgm"]
    missing.extend(command for command in commands if shutil.which(command) is None)
    return missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-latex", action="store_true", help="Check the Text/geometry-only path")
    missing = missing_requirements(parser.parse_args().no_latex)
    print("Missing: " + ", ".join(missing) if missing else "Prerequisites available")
    return int(bool(missing))


if __name__ == "__main__":
    raise SystemExit(main())
