#!/bin/bash

echo "Checking for Python..."
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed. Please install it."
    exit 1
fi

echo "Checking and installing dependencies..."
python3 -m pip install -r requirements.txt

echo "Starting Hermes Dashboard..."
# Try to open browser
if command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:8000 &
elif command -v open &> /dev/null; then
    open http://localhost:8000 &
fi

export PYTHONPATH=$PYTHONPATH:.
python3 dashboard/app.py
