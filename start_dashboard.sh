#!/bin/bash
export HERMES_WEB_DIST=/home/lfdm/.hermes/hermes-agent/hermes_cli/web_dist
cd /home/lfdm/.hermes
exec /home/lfdm/.hermes/hermes-agent/venv/bin/python -c "
import os
os.environ['HERMES_WEB_DIST'] = '/home/lfdm/.hermes/hermes-agent/hermes_cli/web_dist'
from hermes_cli.web_server import start_server
start_server(host='127.0.0.1', port=9119, open_browser=False, allow_public=False)
"
