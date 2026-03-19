
import json
import os
import base64
import numpy as np
from pathlib import Path

import zlib

def decode_ndarray(encoded_value):
    try:
        decoded = base64.b64decode(encoded_value)
        try:
            # Try decompression first
            decompressed = zlib.decompress(decoded)
            return np.frombuffer(decompressed, dtype=np.uint8) # Often uint8 for bit arrays
        except Exception:
            return np.frombuffer(decoded, dtype=np.int64)
    except Exception as e:
        print(f"Decode error: {e}")
        return None

def calculate_entropy(counts):
    total = sum(counts.values())
    if total == 0: return 0
    probs = [v/total for v in counts.values() if v > 0]
    return -sum(p * np.log2(p) for p in probs) / np.log2(len(probs)) if len(probs) > 1 else 0

workload_dir = Path(r"d:\Cosmos\workloads (5)")
showcase_data = []

files = sorted(list(workload_dir.glob("job-*-info.json")))

for info_file in files:
    job_id = info_file.name.split("-")[1]
    result_file = workload_dir / f"job-{job_id}-result.json"
    
    if not result_file.exists():
        continue
        
    with open(info_file, 'r') as f:
        info = json.load(f)
    with open(result_file, 'r') as f:
        result = json.load(f)
        
    # Extract bitstrings from nested Qiskit result structure
    # result['__value__']['pub_results'][0]['__value__']['data']['__value__']['fields']['meas']['__value__']['array']['__value__']
    try:
        data_bin = result['__value__']['pub_results'][0]['__value__']['data']['__value__']
        counts_array_encoded = data_bin['fields']['meas']['__value__']['array']['__value__']
        samples = decode_ndarray(counts_array_encoded)
        
        num_bits = data_bin['fields']['meas']['__value__']['num_bits']
        # Convert integers to bitstrings
        bitstrings = [bin(s)[2:].zfill(num_bits) for s in samples]
        unique_counts = {}
        for b in bitstrings:
            unique_counts[b] = unique_counts.get(b, 0) + 1
            
        entropy = calculate_entropy(unique_counts)
        backend = info.get('backend', 'unknown')
        timestamp = info.get('created', 'unknown')
        
        showcase_data.append({
            "job_id": job_id,
            "backend": backend,
            "timestamp": timestamp,
            "entropy": round(entropy, 4),
            "top_state": max(unique_counts, key=unique_counts.get) if unique_counts else "N/A",
            "total_shots": len(samples)
        })
    except Exception as e:
        print(f"Error processing {job_id}: {e}")

output_path = Path(r"d:\Cosmos\workload_decode_summary.json")
with open(output_path, 'w') as f:
    json.dump(showcase_data, f, indent=2)

print(f"Decoded {len(showcase_data)} jobs. Summary saved to {output_path}")
