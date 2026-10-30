# Keep this a regular package: an installed distribution that ships a top-level
# `scripts` package (gguf does) would otherwise shadow this namespace package
# and break `from scripts.<x> import ...` in scripts/check and friends.
