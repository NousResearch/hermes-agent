import os
import requests

url = "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"
directory = r"c:\Users\corys\The-Cosmic-Davis-12D-Hebbian-Transformer--1\cosmos\cosmos\web\static\js\vendor"
file_path = os.path.join(directory, "chart.min.js")

print(f"Downloading Chart.js from {url}...")
try:
    os.makedirs(directory, exist_ok=True)
    resp = requests.get(url)
    resp.raise_for_status()
    with open(file_path, "wb") as f:
        f.write(resp.content)
    print(f"Chart.js saved to {file_path}")
except Exception as e:
    print(f"Failed to download Chart.js: {e}")
