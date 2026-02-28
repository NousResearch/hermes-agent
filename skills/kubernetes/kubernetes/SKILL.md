---
name: kubernetes
description: Deploy, manage, and debug workloads on Kubernetes clusters using kubectl. Covers pods, deployments, services, ingress, ConfigMaps, secrets, persistent volumes, horizontal scaling, rolling updates, resource inspection, log aggregation, and troubleshooting. Works with any cluster — local (kind, minikube) or cloud (EKS, GKE, AKS). Use when running containerized workloads, deploying ML models, or managing microservice architectures.
version: 1.0.0
author: dogiladeveloper
license: MIT
metadata:
  hermes:
    tags: [Kubernetes, K8s, DevOps, Containers, Deployment, Scaling, kubectl, MLOps, Cloud, EKS, GKE, AKS]
    related_skills: [docker, github-pr-workflow, axolotl, vllm, modal]
    homepage: https://github.com/dogiladeveloper
---

# Kubernetes

Deploy and manage containerized workloads on any Kubernetes cluster.

## Prerequisites

- `kubectl` installed and configured (`kubectl version` to check)
- A valid kubeconfig (`~/.kube/config` or `KUBECONFIG` env var)
- Cluster access (`kubectl cluster-info` to verify)

## Quick Reference

| Action | Command |
|--------|---------|
| List pods | `kubectl get pods` |
| List all resources | `kubectl get all` |
| Describe a resource | `kubectl describe pod <name>` |
| View logs | `kubectl logs <pod>` |
| Exec into pod | `kubectl exec -it <pod> -- bash` |
| Apply manifest | `kubectl apply -f manifest.yaml` |
| Delete resource | `kubectl delete -f manifest.yaml` |
| Scale deployment | `kubectl scale deployment <name> --replicas=3` |
| Rollout status | `kubectl rollout status deployment/<name>` |
| Port forward | `kubectl port-forward pod/<name> 8080:80` |

## Helper Script

This skill includes `scripts/kubectl_manager.py` — a zero-dependency CLI tool
for inspecting cluster health, summarizing workloads, and diagnosing issues.

```bash
python scripts/kubectl_manager.py summary                    # cluster-wide workload overview
python scripts/kubectl_manager.py pods [--namespace NS]      # pod status table
python scripts/kubectl_manager.py nodes                      # node capacity and status
python scripts/kubectl_manager.py events [--namespace NS]    # recent warning events
python scripts/kubectl_manager.py logs <pod> [--namespace NS] [--lines N]
python scripts/kubectl_manager.py top                        # resource usage (requires metrics-server)
python scripts/kubectl_manager.py diagnose <pod> [--namespace NS]  # full pod health report
```

---

## 1. Core Concepts

| Object | Purpose |
|--------|---------|
| **Pod** | Smallest deployable unit — one or more containers sharing network/storage |
| **Deployment** | Manages a set of identical pods; handles rolling updates and rollbacks |
| **Service** | Stable network endpoint for a set of pods (ClusterIP, NodePort, LoadBalancer) |
| **Ingress** | HTTP/HTTPS routing rules to services (requires an ingress controller) |
| **ConfigMap** | Non-secret configuration data injected into pods |
| **Secret** | Sensitive data (passwords, tokens) stored base64-encoded |
| **PersistentVolume** | Cluster-level storage resource |
| **PersistentVolumeClaim** | Request for storage by a pod |
| **Namespace** | Virtual cluster for isolation (default: `default`) |
| **HorizontalPodAutoscaler** | Auto-scales pods based on CPU/memory metrics |

---

## 2. Namespaces

```bash
# List all namespaces
kubectl get namespaces

# Work in a specific namespace (add -n to any command)
kubectl get pods -n kube-system
kubectl get pods -n production

# Set a default namespace for the session (avoid typing -n every time)
kubectl config set-context --current --namespace=production

# List resources across ALL namespaces
kubectl get pods --all-namespaces
kubectl get pods -A   # shorthand
```

---

## 3. Pods

### Viewing pods

```bash
# Basic list
kubectl get pods
kubectl get pods -n staging

# Wide output (shows node, IP)
kubectl get pods -o wide

# Watch live (refreshes automatically)
kubectl get pods -w

# Filter by label
kubectl get pods -l app=myapp
kubectl get pods -l app=myapp,env=production

# Full details
kubectl describe pod myapp-7d4b9c-xk2p9
```

### Running a one-off pod

```bash
# Temporary debug pod (auto-deleted on exit)
kubectl run debug --image=busybox --rm -it --restart=Never -- sh

# Run a specific command
kubectl run curl-test --image=curlimages/curl --rm -it --restart=Never \
  -- curl http://myservice:8080/health
```

### Logs

```bash
# Current logs
kubectl logs myapp-7d4b9c-xk2p9

# Follow (stream)
kubectl logs -f myapp-7d4b9c-xk2p9

# Last N lines
kubectl logs --tail=100 myapp-7d4b9c-xk2p9

# Previous container instance (useful after crash)
kubectl logs myapp-7d4b9c-xk2p9 --previous

# Multi-container pod: specify container
kubectl logs myapp-7d4b9c-xk2p9 -c sidecar

# All pods matching a label (aggregate logs)
kubectl logs -l app=myapp --all-containers=true --prefix=true
```

### Exec into a pod

```bash
# Interactive shell
kubectl exec -it myapp-7d4b9c-xk2p9 -- bash
kubectl exec -it myapp-7d4b9c-xk2p9 -- sh   # for Alpine

# Run a command non-interactively
kubectl exec myapp-7d4b9c-xk2p9 -- env
kubectl exec myapp-7d4b9c-xk2p9 -- cat /etc/config/settings.yaml

# Multi-container pod
kubectl exec -it myapp-7d4b9c-xk2p9 -c worker -- bash
```

---

## 4. Deployments

### Deployment manifest

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
  namespace: production
  labels:
    app: myapp
spec:
  replicas: 3
  selector:
    matchLabels:
      app: myapp
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1         # one extra pod during update
      maxUnavailable: 0   # never take pods down before new ones are ready
  template:
    metadata:
      labels:
        app: myapp
    spec:
      containers:
        - name: myapp
          image: myuser/myapp:v1.2.3
          ports:
            - containerPort: 8000
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: myapp-secrets
                  key: database-url
          resources:
            requests:
              cpu: "100m"
              memory: "128Mi"
            limits:
              cpu: "500m"
              memory: "512Mi"
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 15
            periodSeconds: 30
```

### Managing deployments

```bash
# Apply
kubectl apply -f deployment.yaml

# Check rollout
kubectl rollout status deployment/myapp

# Pause a rollout (useful to canary-test)
kubectl rollout pause deployment/myapp
kubectl rollout resume deployment/myapp

# Rollback to previous version
kubectl rollout undo deployment/myapp

# Rollback to a specific revision
kubectl rollout history deployment/myapp
kubectl rollout undo deployment/myapp --to-revision=3

# Force a restart (pulls new image, same tag)
kubectl rollout restart deployment/myapp

# Scale
kubectl scale deployment myapp --replicas=5

# Update image (triggers rolling update)
kubectl set image deployment/myapp myapp=myuser/myapp:v1.2.4
```

---

## 5. Services

```yaml
# ClusterIP — internal only (default)
apiVersion: v1
kind: Service
metadata:
  name: myapp
spec:
  selector:
    app: myapp
  ports:
    - port: 80
      targetPort: 8000
  type: ClusterIP

---
# LoadBalancer — external (cloud provider provisions LB)
apiVersion: v1
kind: Service
metadata:
  name: myapp-external
spec:
  selector:
    app: myapp
  ports:
    - port: 80
      targetPort: 8000
  type: LoadBalancer
```

```bash
# List services
kubectl get services
kubectl get svc   # shorthand

# Describe (shows endpoints, selector)
kubectl describe svc myapp

# Quick port-forward to test locally (no service needed)
kubectl port-forward deployment/myapp 8080:8000
kubectl port-forward svc/myapp 8080:80
```

---

## 6. ConfigMaps & Secrets

### ConfigMaps

```bash
# Create from literal values
kubectl create configmap myapp-config \
  --from-literal=LOG_LEVEL=info \
  --from-literal=MAX_WORKERS=4

# Create from a file
kubectl create configmap myapp-config --from-file=config.yaml

# View
kubectl get configmap myapp-config -o yaml

# Update in-place
kubectl edit configmap myapp-config
```

Use in a pod:

```yaml
envFrom:
  - configMapRef:
      name: myapp-config

# Or mount as files
volumes:
  - name: config
    configMap:
      name: myapp-config
volumeMounts:
  - name: config
    mountPath: /etc/config
```

### Secrets

```bash
# Create secret (values are base64-encoded automatically)
kubectl create secret generic myapp-secrets \
  --from-literal=database-url="postgres://user:pass@db:5432/app" \
  --from-literal=api-key="sk-secret123"

# Create from a .env file
kubectl create secret generic myapp-secrets --from-env-file=.env

# View (values are base64 — decode to read)
kubectl get secret myapp-secrets -o yaml
kubectl get secret myapp-secrets -o jsonpath='{.data.database-url}' | base64 --decode

# Update a single key
kubectl patch secret myapp-secrets \
  -p '{"data":{"api-key":"'$(echo -n "new-secret" | base64)'"}}'
```

---

## 7. Persistent Storage

```yaml
# PersistentVolumeClaim
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: myapp-data
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
  storageClassName: standard   # depends on your cluster

---
# Use in a pod
volumes:
  - name: data
    persistentVolumeClaim:
      claimName: myapp-data
volumeMounts:
  - name: data
    mountPath: /data
```

```bash
kubectl get pvc
kubectl get pv
kubectl describe pvc myapp-data
```

---

## 8. Horizontal Pod Autoscaling

```bash
# Autoscale based on CPU (requires metrics-server)
kubectl autoscale deployment myapp \
  --min=2 --max=10 --cpu-percent=70

# Check HPA status
kubectl get hpa
kubectl describe hpa myapp
```

```yaml
# Declarative HPA manifest
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: myapp
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: myapp
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
```

---

## 9. Resource Management

```bash
# View resource usage (requires metrics-server)
kubectl top nodes
kubectl top pods
kubectl top pods -n production --sort-by=memory

# View resource requests/limits set on pods
kubectl get pods -o=custom-columns=\
'NAME:.metadata.name,CPU-REQ:.spec.containers[*].resources.requests.cpu,MEM-REQ:.spec.containers[*].resources.requests.memory,CPU-LIM:.spec.containers[*].resources.limits.cpu,MEM-LIM:.spec.containers[*].resources.limits.memory'

# ResourceQuota — limit total resources in a namespace
kubectl get resourcequota
kubectl describe resourcequota
```

```yaml
# Namespace resource quota
apiVersion: v1
kind: ResourceQuota
metadata:
  name: production-quota
  namespace: production
spec:
  hard:
    requests.cpu: "4"
    requests.memory: 8Gi
    limits.cpu: "8"
    limits.memory: 16Gi
    pods: "20"
```

---

## 10. Ingress

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: myapp-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  ingressClassName: nginx
  rules:
    - host: myapp.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: myapp
                port:
                  number: 80
          - path: /api
            pathType: Prefix
            backend:
              service:
                name: myapp-api
                port:
                  number: 8000
  tls:
    - hosts:
        - myapp.example.com
      secretName: myapp-tls-cert
```

```bash
kubectl get ingress
kubectl describe ingress myapp-ingress
```

---

## 11. Troubleshooting

### Pod won't start — checklist

```bash
# Step 1: Check pod status and events
kubectl describe pod <pod-name>
# Look at: Events section at the bottom — this shows exactly what went wrong

# Step 2: Check logs (even for crashed pods)
kubectl logs <pod-name>
kubectl logs <pod-name> --previous   # logs from before the crash

# Step 3: Check node pressure
kubectl describe node <node-name>
# Look for: Conditions (MemoryPressure, DiskPressure, PIDPressure)

# Step 4: Check image pull
kubectl get events --sort-by=.lastTimestamp | grep -i "failed\|error\|back-off"
```

### Common error messages

| Error | Cause | Fix |
|-------|-------|-----|
| `ImagePullBackOff` | Cannot pull image | Check image name/tag, registry credentials |
| `CrashLoopBackOff` | Container keeps crashing | Check logs with `--previous`, fix app error |
| `Pending` (no node) | Insufficient resources | Scale cluster or reduce resource requests |
| `OOMKilled` | Out of memory | Increase memory limit or fix memory leak |
| `ErrImageNeverPull` | Image policy prevents pull | Change `imagePullPolicy` or push image |
| `CreateContainerConfigError` | Missing ConfigMap/Secret | Create the referenced resource |

### Useful diagnostic commands

```bash
# All recent warning events in a namespace
kubectl get events --field-selector type=Warning -n production --sort-by=.lastTimestamp

# Which node is a pod on?
kubectl get pod myapp-7d4b9c-xk2p9 -o wide

# Is the service actually routing to pods?
kubectl get endpoints myapp
# If ENDPOINTS shows <none>, the service selector doesn't match any pods

# Check DNS from inside a pod
kubectl exec -it myapp-7d4b9c-xk2p9 -- nslookup myservice
kubectl exec -it myapp-7d4b9c-xk2p9 -- curl http://myservice/health

# Check if resource limits are being hit
kubectl describe pod myapp-7d4b9c-xk2p9 | grep -A5 "Limits\|Requests\|Last State"
```

---

## 12. Local Development Clusters

### kind (Kubernetes in Docker)

```bash
# Install
brew install kind   # macOS
# or: curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.23.0/kind-linux-amd64 && chmod +x ./kind

# Create cluster
kind create cluster --name dev

# Create with custom config (multi-node)
cat <<EOF | kind create cluster --name dev --config=-
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
  - role: worker
  - role: worker
EOF

# Load a local Docker image into kind (no registry needed)
kind load docker-image myapp:latest --name dev

# Delete cluster
kind delete cluster --name dev
```

### minikube

```bash
# Start cluster
minikube start --cpus=4 --memory=8192

# Use local Docker images (no push needed)
eval $(minikube docker-env)
docker build -t myapp:latest .

# Access services
minikube service myapp --url

# Enable addons
minikube addons enable ingress
minikube addons enable metrics-server

# Stop / delete
minikube stop
minikube delete
```

---

## 13. Deploying ML Models

Example: deploying a vLLM inference server.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-server
  namespace: ml-inference
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm-server
  template:
    metadata:
      labels:
        app: vllm-server
    spec:
      containers:
        - name: vllm
          image: vllm/vllm-openai:latest
          args:
            - --model
            - meta-llama/Llama-3-8B-Instruct
            - --port
            - "8000"
          ports:
            - containerPort: 8000
          resources:
            limits:
              nvidia.com/gpu: 1       # requires NVIDIA device plugin
          env:
            - name: HUGGING_FACE_HUB_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-secrets
                  key: token
          volumeMounts:
            - name: model-cache
              mountPath: /root/.cache/huggingface
      volumes:
        - name: model-cache
          persistentVolumeClaim:
            claimName: model-cache-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-server
  namespace: ml-inference
spec:
  selector:
    app: vllm-server
  ports:
    - port: 8000
      targetPort: 8000
  type: ClusterIP
```

```bash
# Deploy
kubectl apply -f vllm-deployment.yaml

# Check GPU allocation
kubectl describe node | grep -A5 "nvidia.com/gpu"

# Test inference
kubectl port-forward svc/vllm-server 8000:8000 &
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3-8B-Instruct", "prompt": "Hello", "max_tokens": 50}'
```

---

## Contributing

Skill authored by **dogiladeveloper**.

- GitHub: [github.com/dogiladeveloper](https://github.com/dogiladeveloper)
- Discord: `dogiladeveloper`
- Twitter/X: [@dogiladeveloper](https://twitter.com/dogiladeveloper)

Issues, improvements, and pull requests are welcome!
