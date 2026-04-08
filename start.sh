#!/bin/bash

echo "Checking for Python..."
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed. Please install it."
    exit 1
fi

echo "Checking dependencies..."
python3 -c "import fastapi, uvicorn, jinja2" &> /dev/null
if [ $? -ne 0 ]; then
    echo "Installing missing dependencies..."
    python3 -m pip install fastapi uvicorn jinja2 pydantic
fi

echo "Starting Hermes Dashboard..."
# Try to open browser
if command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:8000 &
elif command -v open &> /dev/null; then
    open http://localhost:8000 &
fi

export PYTHONPATH=$PYTHONPATH:.
python3 dashboard/app.py
