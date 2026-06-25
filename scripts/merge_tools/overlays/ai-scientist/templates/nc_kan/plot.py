import json
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np

labels = {"run_0": "Baseline"}
folders = [f for f in os.listdir("./") if f.startswith("run") and osp.isdir(f)]
for folder in folders:
    with open(osp.join(folder, "final_info.json"), "r", encoding="utf-8") as handle:
        data = json.load(handle)
    means = data["nc_kan"]["means"]
    keys = ["accuracy", "nc_ratio", "class_separation"]
    values = [means[k] for k in keys]
    plt.figure(figsize=(8, 4))
    plt.bar(keys, values, color=["#4C78A8", "#F58518", "#54A24B"])
    plt.title(f"nc_kan metrics ({labels.get(folder, folder)})")
    plt.tight_layout()
    plt.savefig(f"nc_kan_metrics_{folder}.png")
    plt.close()
