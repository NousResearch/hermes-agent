build-base:
	docker build -t hermes-agent-base:latest -f Dockerfile.base .

build-app:
	docker build -t hermes-agent:latest .

.PHONY: build-base build-app
