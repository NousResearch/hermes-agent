# Agent Sandbox terminal backend

This bundled backend runs Hermes terminal and file operations in one disposable
Agent Sandbox task. It requires:

- an absolute `kubectl_path` in the plugin settings;
- the fixed `agent-sandbox-tasks` namespace;
- an immutable coding image digest;
- a namespace-scoped operator identity.

The adapter creates one labelled `Sandbox`, waits for its task Pod, and runs
commands with `kubectl exec`. The task Pod has no Kubernetes service-account
token. The adapter deletes the Sandbox after cleanup and reports cleanup
failures.

Configure the backend in `config.yaml` through the plugin settings path:

```yaml
plugins:
  entries:
    terminal/agent_sandbox:
      settings:
        backend:
          kubectl_path: /usr/local/bin/kubectl
          namespace: agent-sandbox-tasks
          image: registry.example/hermes-coding@sha256:<64-hex-digest>
          deadline: 600
          ready_timeout: 120
          cleanup_timeout: 60
          command_timeout: 600
terminal:
  backend: agent_sandbox
```

The adapter does not accept a task image, namespace, kubeconfig path, or
resource name from a terminal command. Private-repository credentials are not
injected by this backend. Configure and verify a separate reviewed credential
path before using a task for private checkout or Forgejo publication.

The offline contract tests are in
`tests/plugins/test_agent_sandbox_provider.py`. A live smoke test requires
current, action-specific operator approval.
