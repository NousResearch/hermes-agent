FROM ghcr.io/vectorize-io/hindsight-api@sha256:7635a15739361dbdf221ba796ad25a813f876144fe113022eea8e26cb6ee75e7
USER root
COPY docs/specs/reference/hindsight-config.json /opt/employee/hindsight-config.json
COPY deploy/railway/hindsight/ /opt/employee/
USER hindsight
ENTRYPOINT ["python", "/opt/employee/start.py"]
