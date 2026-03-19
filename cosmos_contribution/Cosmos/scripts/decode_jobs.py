import json
from qiskit_ibm_runtime.utils.json import RuntimeDecoder

def analyze_jobs():
    info_file = r"D:\Cosmos\Cosmos\workloads\workloads (2)\job-d6lmi8u9td6c73am3l60-info.json"
    res_file = r"D:\Cosmos\Cosmos\workloads\workloads (2)\job-d6lmi8u9td6c73am3l60-result.json"

    with open(info_file, 'r') as f:
        info = json.load(f, cls=RuntimeDecoder)
    
    circuit = info["params"]["pubs"][0][0]
    print("================== QASM CIRCUIT ==================")
    try:
        print(circuit.qasm())
    except:
        print(circuit)
    
    with open(res_file, 'r') as f:
        res = json.load(f, cls=RuntimeDecoder)
        
    print("\n================== MEASUREMENT DATA ==================")
    try:
        # For V2 Primitives, it's PrimitiveResult. Iterating over it gives PubResults
        for i, pub in enumerate(res):
            print(f"--- Pub {i} ---")
            counts = pub.data.meas.get_counts()
            print("COUNTS:", counts)
    except Exception as e:
        print(f"Error getting counts: {e}")
        print("Dir of res:", dir(res))
        
if __name__ == "__main__":
    analyze_jobs()
