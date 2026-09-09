#!/usr/bin/env bash
# Termux Installation Script for Hermes Agent Cross-Platform V2

set -e

echo "=== Hermes Agent Termux Setup ==="

pkg update -y && pkg upgrade -y
pkg install -y python git clang libffi openssl termux-api

pip install --upgrade pip
if [ -f "constraints-termux.txt" ]; then
    pip install -r constraints-termux.txt
fi

pip install -e .

echo "=== Termux Setup Complete ==="
echo "Run 'hermes doctor' to verify installation."
