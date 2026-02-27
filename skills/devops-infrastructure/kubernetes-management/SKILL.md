---
name: kubernetes-management
description: Deploy, manage, and monitor Kubernetes clusters, pods, services, and Helm releases using kubectl. View logs, exec into containers, manage contexts, and troubleshoot cluster issues.
version: 1.0.0
author: community
license: MIT
metadata:
  hermes:
    tags: [Kubernetes, k8s, kubectl, Helm, DevOps, Containers, Orchestration]
    related_skills: [docker-orchestration, terraform-iac, monitoring-logging]
---

# Kubernetes Management

Manage Kubernetes clusters, resources, and deployments via kubectl. Works with any Kubernetes cluster (EKS, GKE, AKS, minikube, k3s, etc.).

## Prerequisites

- kubectl installed (`command -v kubectl`)
- Valid kubeconfig (`~/.kube/config` or `$KUBECONFIG`)
- Cluster access configured

### Setup

```bash
# Verify kubectl is installed
kubectl version --client

# Check current context
kubectl config current-context

# List all contexts
kubectl config get-contexts

# Switch context
kubectl config use-context <context-name>

# Set namespace for current context
kubectl config set-context --current --namespace=<namespace>
```

---

## 1. Viewing Resources

### Pods

```bash
# List pods in current namespace
kubectl get pods

# List pods in all namespaces
kubectl get pods --all-namespaces
kubectl get pods -A

# Wide output with node names
kubectl get pods -o wide

# Watch pods in real-time
kubectl get pods -w

# Get pod details
kubectl describe pod <pod-name>

# Get pod YAML
kubectl get pod <pod-name> -o yaml

# Get pod logs
kubectl logs <pod-name>

# Follow logs in real-time
kubectl logs -f <pod-name>

# Previous container logs (for crashed pods)
kubectl logs <pod-name> --previous

# Logs from specific container in multi-container pod
kubectl logs <pod-name> -c <container-name>
```

### Deployments

```bash
# List deployments
kubectl get deployments
kubectl get deploy

# Deployment details
kubectl describe deployment <deployment-name>

# Deployment rollout status
kubectl rollout status deployment/<deployment-name>

# Deployment history
kubectl rollout history deployment/<deployment-name>

# Rollback to previous revision
kubectl rollout undo deployment/<deployment-name>

# Rollback to specific revision
kubectl rollout undo deployment/<deployment-name> --to-revision=2

# Scale deployment
kubectl scale deployment/<deployment-name> --replicas=5

# Autoscale deployment
kubectl autoscale deployment/<deployment-name> --min=2 --max=10 --cpu-percent=80
```

### Services

```bash
# List services
kubectl get services
kubectl get svc

# Service details
kubectl describe service <service-name>

# Get service endpoints
kubectl get endpoints <service-name>
```

### Nodes

```bash
# List nodes
kubectl get nodes

# Node details
kubectl describe node <node-name>

# Node resource usage
kubectl top nodes
```

### All Resources

```bash
# List all resources in current namespace
kubectl get all

# List specific resource types
kubectl get pods,svc,deploy,ingress

# List events
kubectl get events
kubectl get events --field-selector type=Warning
kubectl get events --sort-by='.lastTimestamp'
```

---

## 2. Creating Resources

### From YAML/JSON Files

```bash
# Apply a configuration file
kubectl apply -f manifest.yaml

# Apply from URL
kubectl apply -f https://example.com/manifest.yaml

# Apply multiple files
kubectl apply -f dir/
kubectl apply -f manifest1.yaml -f manifest2.yaml

# Dry run (validate without applying)
kubectl apply -f manifest.yaml --dry-run=client
kubectl apply -f manifest.yaml --dry-run=server
```

### Quick Pod Creation

```bash
# Run a simple pod
kubectl run nginx --image=nginx

# Run with specific namespace
kubectl run nginx --image=nginx -n my-namespace

# Run and expose
kubectl run nginx --image=nginx --port=80 --expose

# Run with labels
kubectl run nginx --image=nginx --labels="app=web,tier=frontend"

# Run with environment variables
kubectl run nginx --image=nginx --env="KEY=value"

# Run with resource limits
kubectl run nginx --image=nginx --limits="cpu=500m,memory=512Mi"

# Run and override command
kubectl run debug --image=busybox --restart=Never -- sleep 3600

# Run interactive pod
kubectl run -it debug --image=busybox --restart=Never -- sh
```

### Create from Command Line

```bash
# Create deployment
kubectl create deployment nginx --image=nginx --replicas=3

# Create service
kubectl create service clusterip my-service --tcp=80:8080

# Create namespace
kubectl create namespace my-namespace

# Create configmap
kubectl create configmap my-config --from-literal=key=value
kubectl create configmap my-config --from-file=path/to/file

# Create secret
kubectl create secret generic my-secret --from-literal=password=secret123
kubectl create secret tls my-tls-secret --cert=path/to/cert.pem --key=path/to/key.pem

# Create serviceaccount
kubectl create serviceaccount my-sa
```

---

## 3. Executing Commands in Containers

```bash
# Execute command in pod
kubectl exec <pod-name> -- <command>

# Interactive shell
kubectl exec -it <pod-name> -- /bin/sh
kubectl exec -it <pod-name> -- /bin/bash

# Specific container in multi-container pod
kubectl exec -it <pod-name> -c <container-name> -- /bin/sh

# Run command with environment
kubectl exec <pod-name> -- env

# Copy files to/from pod
kubectl cp <pod-name>:/path/in/pod /local/path
kubectl cp /local/path <pod-name>:/path/in/pod
kubectl cp <pod-name>:/path/in/pod <pod2-name>:/path/in/pod2
```

---

## 4. Port Forwarding

```bash
# Forward local port to pod port
kubectl port-forward <pod-name> 8080:80

# Forward to specific address
kubectl port-forward --address 0.0.0.0 <pod-name> 8080:80

# Forward to service
kubectl port-forward svc/<service-name> 8080:80

# Forward to deployment
kubectl port-forward deploy/<deployment-name> 8080:80

# Background port forwarding
kubectl port-forward <pod-name> 8080:80 &
```

---

## 5. Debugging and Troubleshooting

### Pod Debugging

```bash
# Describe pod (events, conditions)
kubectl describe pod <pod-name>

# Get pod logs
kubectl logs <pod-name>

# Previous container logs (crashed pods)
kubectl logs <pod-name> --previous

# Stream logs
kubectl logs -f <pod-name>

# Logs since specific time
kubectl logs <pod-name> --since=1h

# Logs with timestamps
kubectl logs <pod-name> --timestamps

# Execute debug shell
kubectl exec -it <pod-name> -- /bin/sh

# Debug with ephemeral container (k8s 1.23+)
kubectl debug -it <pod-name> --image=busybox --target=<container>

# Copy pod for debugging
kubectl debug <pod-name> -it --copy-to=debug-pod --container=<container> -- sh
```

### Common Issues

```bash
# Pods stuck in Pending
kubectl describe pod <pod-name>  # Check events for resource/affinity issues
kubectl get nodes  # Check node capacity

# Pods stuck in CrashLoopBackOff
kubectl logs <pod-name> --previous  # Check crash logs
kubectl describe pod <pod-name>  # Check restart count and events

# ImagePullBackOff
kubectl describe pod <pod-name>  # Check image name and pull secrets

# OOMKilled
kubectl describe pod <pod-name>  # Check memory limits
kubectl top pod <pod-name>  # Check actual usage

# High resource usage
kubectl top nodes
kubectl top pods
kubectl top pods --all-namespaces
```

---

## 6. Helm Operations

```bash
# Add a chart repository
helm repo add <name> <url>
helm repo add stable https://charts.helm.sh/stable

# List repositories
helm repo list

# Update repositories
helm repo update

# Search for charts
helm search repo <keyword>
helm search hub <keyword>

# Install a chart
helm install <release-name> <chart>
helm install my-nginx stable/nginx-ingress

# Install with values
helm install <release-name> <chart> --set key=value
helm install <release-name> <chart> -f values.yaml

# List releases
helm list
helm list --all-namespaces

# Get release status
helm status <release-name>

# Get release values
helm get values <release-name>
helm get values <release-name> --all

# Upgrade release
helm upgrade <release-name> <chart>
helm upgrade <release-name> <chart> --set key=value

# Rollback release
helm rollback <release-name> <revision>

# Uninstall release
helm uninstall <release-name>
helm delete <release-name>

# History
helm history <release-name>

# Template rendering (dry-run)
helm template <chart>
helm template <chart> --set key=value
```

---

## 7. Namespace Management

```bash
# List namespaces
kubectl get namespaces
kubectl get ns

# Create namespace
kubectl create ns <namespace-name>

# Delete namespace (and all resources)
kubectl delete ns <namespace-name>

# Set default namespace for context
kubectl config set-context --current --namespace=<namespace>

# Run command in specific namespace
kubectl get pods -n <namespace>
kubectl -n <namespace> get pods

# Get all resources in namespace
kubectl api-resources --verbs=list --namespaced -o name | xargs -n 1 kubectl get --show-kind --ignore-not-found -n <namespace>
```

---

## 8. Resource Management

### Labels and Selectors

```bash
# Add label to pod
kubectl label pod <pod-name> env=production

# Remove label
kubectl label pod <pod-name> env-

# Overwrite label
kubectl label pod <pod-name> env=development --overwrite

# List pods with selector
kubectl get pods -l env=production
kubectl get pods -l 'env in (production, staging)'
kubectl get pods -l 'app=web,tier=frontend'

# List all labels
kubectl get pods --show-labels
```

### Annotations

```bash
# Add annotation
kubectl annotate pod <pod-name> description="my annotation"

# Remove annotation
kubectl annotate pod <pod-name> description-
```

### Resource Quotas and Limits

```bash
# Get resource quotas
kubectl get resourcequota

# Get limit ranges
kubectl get limitrange

# Describe to see constraints
kubectl describe resourcequota <name>
```

---

## 9. Cluster Management

```bash
# Cluster info
kubectl cluster-info
kubectl cluster-info dump

# API resources
kubectl api-resources
kubectl api-resources --namespaced=true
kubectl api-resources --verbs=list,get

# API versions
kubectl api-versions

# Get all CRDs
kubectl get crd
kubectl get crd <crd-name> -o yaml

# Certificate management
kubectl get certificates
kubectl get cert  # shorthand
kubectl get cert --all-namespaces

# View kubeconfig
kubectl config view
kubectl config view --raw

# Current context
kubectl config current-context

# Use context
kubectl config use-context <context>

# Rename context
kubectl config rename-context <old-name> <new-name>

# Delete context
kubectl config delete-context <name>
```

---

## 10. Useful Shortcuts

```bash
# Aliases (add to ~/.bashrc or ~/.zshrc)
alias k='kubectl'
alias kgp='kubectl get pods'
alias kgs='kubectl get svc'
alias kgd='kubectl get deploy'
alias kgn='kubectl get nodes'
alias kgns='kubectl get ns'
alias kd='kubectl describe'
alias kl='kubectl logs'
alias klf='kubectl logs -f'
alias ke='kubectl exec'
alias kex='kubectl exec -it'
alias kpf='kubectl port-forward'
alias kctx='kubectl config use-context'
alias kns='kubectl config set-context --current --namespace'

# kubectl plugins (via krew)
kubectl krew install ctx
kubectl krew install ns
kubectl krew install tail

# Then use:
kubectl ctx <context-name>
kubectl ns <namespace-name>
kubectl tail <pod-name>
```

---

## 11. Advanced Operations

### JSONPath Queries

```bash
# Get specific fields
kubectl get pods -o jsonpath='{.items[*].metadata.name}'
kubectl get pods -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.phase}{"\n"}{end}'

# Get node IPs
kubectl get nodes -o jsonpath='{.items[*].status.addresses[?(@.type=="InternalIP")].address}'
```

### Custom Columns

```bash
# Custom output format
kubectl get pods -o custom-columns='NAME:.metadata.name,STATUS:.status.phase,NODE:.spec.nodeName'

# Wide output with specific fields
kubectl get pods -o custom-columns='POD:.metadata.name,RESTARTS:.status.containerStatuses[0].restartCount'
```

### Dry Run and Diff

```bash
# Server-side dry run
kubectl apply -f manifest.yaml --dry-run=server

# Diff current vs desired
kubectl diff -f manifest.yaml
```

### Watch with Conditions

```bash
# Wait for condition
kubectl wait --for=condition=ready pod <pod-name> --timeout=60s
kubectl wait --for=condition=available deployment/<deployment-name>

# Watch with selector
kubectl get pods -l app=web -w
```

---

## Security Note

Always follow principle of least privilege:
- Use service accounts with minimal permissions
- Avoid using cluster-admin for daily operations
- Enable audit logging
- Rotate credentials regularly
- Use network policies to restrict traffic
