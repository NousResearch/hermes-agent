FROM debian

RUN apt update
RUN apt install -y nodejs npm python3 python3-pip ripgrep ffmpeg gcc python3-dev libffi-dev

COPY . /opt/hermes
WORKDIR /opt/hermes

RUN pip install -e ".[all]" --break-system-packages
RUN pip install -e "./mini-swe-agent" --break-system-packages
# RUN pip install -e "./tinker-atropos" --break-system-packages
RUN npm install
RUN npx playwright install --with-deps chromium

RUN chmod +x /opt/hermes/docker/entrypoint.sh

ENV HERMES_HOME=/opt/data
VOLUME [ "/opt/data" ]
ENTRYPOINT [ "/opt/hermes/docker/entrypoint.sh" ]