import json
import matplotlib.pyplot as plt
from pathlib import Path
from qiskit.visualization import plot_histogram
from qiskit_ibm_runtime.utils.json import RuntimeDecoder
import datetime

def process_all_jobs():
    workload_dir = Path(r"D:\Cosmos\Cosmos\workloads\workloads (2)")
    artifact_dir = Path(r"C:\Users\corys\.gemini\antigravity\brain\e39fa9c0-1407-4d84-9683-4edbfa79866b")
    
    info_files = list(workload_dir.glob("*-info.json"))
    md_path = artifact_dir / "all_jobs_breakdown.md"
    
    with open(md_path, "w", encoding="utf-8") as md:
        md.write("# Exhaustive Swarm Quantum Execution Breakdown\n\n")
        md.write("This document contains **ALL** telemetry data extracted from the 19 IBM Quantum primitives executed by the Swarm. It includes raw parameters, full measurement tables, backend metadata, and circuit schema.\n\n")
        
        for idx, info_file in enumerate(info_files):
            try:
                job_id = info_file.name.replace("job-", "").replace("-info.json", "")
                md.write(f"## 🛸 Job #{idx+1} | Transation ID: `{job_id}`\n")
                
                # Load QASM / Info
                with open(info_file, 'r') as f:
                    info = json.load(f, cls=RuntimeDecoder)
                
                creation = info.get("creation_date", "Unknown")
                backend = info.get("backend", "Unknown")
                status = info.get("status", "Unknown")
                
                md.write("### Telemetry Metadata\n")
                md.write(f"- **Execution Timestamp:** `{creation}`\n")
                md.write(f"- **Quantum Backend:** `{backend}`\n")
                md.write(f"- **Final Status:** `{status}`\n\n")
                
                pubs = info.get("params", {}).get("pubs", [])
                if len(pubs) > 0:
                    circuit = pubs[0][0]
                    params = pubs[0][1] if len(pubs[0]) > 1 else None
                    shots = pubs[0][2] if len(pubs[0]) > 2 else 4096
                    
                    md.write(f"- **Dimensions / Qubits:** `{circuit.num_qubits}`\n")
                    md.write(f"- **Target Shots:** `{shots}`\n\n")
                    
                    if params is not None:
                        md.write("### Dimensional Parameter Bindings\n")
                        md.write("```json\n")
                        md.write(str(params) + "\n")
                        md.write("```\n\n")
                    
                    qasm_txt = str(circuit.draw(output='text'))
                    md.write("### 12D Phase Cognitive Circuit (QASM)\n")
                    md.write(f"```text\n{qasm_txt}\n```\n\n")
                
                # Handle Results
                res_file = workload_dir / info_file.name.replace("-info", "-result")
                if not res_file.exists():
                    md.write("> *Job measurement result JSON missing from directory.*\n\n---\n\n")
                    continue
                    
                with open(res_file, 'r') as f:
                    res = json.load(f, cls=RuntimeDecoder)
                    
                # Full Results
                counts = res[0].data.meas.get_counts()
                sorted_counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))
                
                # Usage metadata from result
                # primitive_res may have metadata on the root or pub_results
                meta = res[0].metadata if hasattr(res[0], 'metadata') else {}
                if meta:
                    md.write("### Execution Profiling\n")
                    for k,v in meta.items():
                        md.write(f"- **{k}:** `{v}`\n")
                    md.write("\n")
                
                img_name = f"hist_{job_id}.png"
                img_path = artifact_dir / img_name
                
                # Plot top 15 for aesthetics
                fig = plot_histogram(dict(list(sorted_counts.items())[:15]), title=f"Execution Entropy (Job {job_id})", color='#2eb82e', figsize=(10,5))
                plt.savefig(img_path, dpi=200, bbox_inches='tight')
                plt.close(fig)
                
                md.write("### Measurement Entropy Collapse Graph\n")
                md.write(f"![Histogram]({img_name})\n\n")
                
                md.write("### Exhaustive Vector State Distribution\n")
                md.write("Every physical state string the universal probability collapsed into:\n\n")
                md.write("| Measured Bitstring | Decoherence Hits (Raw Shots) |\n")
                md.write("| :--- | :--- |\n")
                for state, hits in sorted_counts.items():
                    md.write(f"| `|{state}⟩` | {hits} |\n")
                
                md.write("\n---\n\n")
                
            except Exception as e:
                md.write(f"> *Error decoding this specific job trace: {e}*\n\n---\n\n")

    print(f"SUCCESS! Wrote all exhaustive jobs into {md_path}")

if __name__ == "__main__":
    process_all_jobs()
