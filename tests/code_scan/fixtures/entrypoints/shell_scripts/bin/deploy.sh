#!/bin/bash
# Deploy script
set -e

echo "Deploying application..."
python manage.py migrate
python manage.py collectstatic --noinput
echo "Deployment complete."
