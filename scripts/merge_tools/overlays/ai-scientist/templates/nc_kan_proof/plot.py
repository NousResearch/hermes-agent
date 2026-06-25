import json
import os
import os.path as osp

import matplotlib.pyplot as plt

for folder in [f for f in os.listdir("./") if f.startswith("run") and osp.isdir(f)]:
    with open(osp.join(folder, "final_info.json"), "r", encoding="utf-8") as handle:
        means = json.load(handle)["nc_kan_proof"]["means"]
    keys = list(means.keys())
    vals = [means[k] for k in keys]
    plt.figure(figsize=(7, 4))
    plt.bar(keys, vals, color="#72B7B2")
    plt.title(f"nc_kan_proof ({folder})")
    plt.tight_layout()
    plt.savefig(f"nc_kan_proof_{folder}.png")
    plt.close()
