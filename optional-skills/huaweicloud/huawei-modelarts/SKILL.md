---
name: huawei-modelarts
description: 'Use when training, deploying, or managing AI/ML models on Huawei Cloud ModelArts. Covers training jobs, model registry, online services, notebook instances. Triggers: ModelArts, model training, AI, machine learning, deep learning, notebook, inference, deployment. NOT for: general AI/ML concepts, non-Huawei platforms.'
version: 1
---

# Huawei Cloud ModelArts

**STOP - Do not answer from general knowledge.** Follow the procedure below.

Always run `hcloud ModelArts <Operation> --help` before constructing commands to discover exact parameter names and requirements.

## Overview

Domain expertise for ModelArts. Covers training jobs, model deployment, notebook instances, and OBS integration.

## Critical Warnings

| Trap                            | Why                                                                                 |
| ------------------------------- | ----------------------------------------------------------------------------------- |
| OBS bucket required             | All training data, model outputs, and notebook storage use OBS. Create bucket first |
| Training charges by duration    | Pay-per-minute GPU/CPU billing. Stop unused notebooks and services                  |
| Notebook auto-stop needed       | Default no auto-stop — can run indefinitely and incur charges                       |
| Model deployment needs quota    | Online services may require service quota approval in new accounts                  |
| Training job output must be OBS | Local output not supported. Ensure `--output_path` is a valid OBS path              |

## Prerequisites

- Training data must be stored in an OBS bucket (see `huawei-obs`)
- Output path for trained models must be an OBS path
- Notebook instances need a VPC/subnet (see `huawei-vpc`)

## Common Workflows

| Task               | Operation                                          |
| ------------------ | -------------------------------------------------- |
| List models        | `ListModels --cli-region=<r> --project_id=<p>`     |
| List training jobs | `ListTrainJobs --cli-region=<r> --project_id=<p>`  |
| List services      | `ListServices --cli-region=<r> --project_id=<p>`   |
| List notebooks     | `ListNotebooks --cli-region=<r> --project_id=<p>`  |
| Create model       | `CreateModel --cli-region=<r> --project_id=<p>`    |
| Create notebook    | `CreateNotebook --cli-region=<r> --project_id=<p>` |

Discover operation parameters with `--help` before executing any write operation.

## Service Types

| Type             | Purpose                                 |
| ---------------- | --------------------------------------- |
| Training Job     | Train models on GPU/CPU resources       |
| Model            | Register trained models for deployment  |
| Service (Online) | Deploy models as REST API endpoints     |
| Notebook         | JupyterLab environments for development |

## Troubleshooting

| Error              | Fix                                                              |
| ------------------ | ---------------------------------------------------------------- |
| OBS path not found | Verify OBS bucket exists and path is correct                     |
| Training job fails | Check training data format, resource quotas, and logs in console |

## Cross-Skill References

- **OBS storage**: See `huawei-obs` for bucket and training data management
- **VPC/Subnet**: See `huawei-vpc` for notebook network configuration
