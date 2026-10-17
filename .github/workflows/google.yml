# This workflow will build a docker container, publish it to Google Container
# Registry, and deploy it to GKE when there is a push to the "main"
# branch.
#
# To configure this workflow:
#
# 1. Enable the following Google Cloud APIs:
#
#    - Artifact Registry (artifactregistry.googleapis.com)
#    - Google Kubernetes Engine (container.googleapis.com)
#    - IAM Credentials API (iamcredentials.googleapis.com)
#
#    You can learn more about enabling APIs at
#    https://support.google.com/googleapi/answer/6158841.
#
# 2. Ensure that your repository contains the necessary configuration for your
#    Google Kubernetes Engine cluster, including deployment.yml,
#    kustomization.yml, service.yml, etc.
#
# 3. Create and configure a Workload Identity Provider for GitHub:
#    https://github.com/google-github-actions/auth#preferred-direct-workload-identity-federation.
#
#    Depending on how you authenticate, you will need to grant an IAM principal
#    permissions on Google Cloud:
#
#    - Artifact Registry Administrator (roles/artifactregistry.admin)
#    - Kubernetes Engine Developer (roles/container.developer)
#
#    You can learn more about setting IAM permissions at
#    https://cloud.google.com/iam/docs/manage-access-other-resources
#
# 5. Change the values in the "env" block to match your values.

name: 'Build and Deploy to GKE'

on:
  push:
    branches:
      - '"main"'

env:
  PROJECT_ID: 'my-project' # TODO: update to your Google Cloud project ID
  GAR_LOCATION: 'us-central1' # TODO: update to your region
  GKE_CLUSTER: 'cluster-1' # TODO: update to your cluster name
  GKE_ZONE: 'us-central1-c' # TODO: update to your cluster zone
  DEPLOYMENT_NAME: 'gke-test' # TODO: update to your deployment name
  REPOSITORY: 'samples' # TODO: update to your Artifact Registry docker repository name
  IMAGE: 'static-site'
  WORKLOAD_IDENTITY_PROVIDER: 'projects/123456789/locations/global/workloadIdentityPools/my-pool/providers/my-provider' # TODO: update to your workload identity provider

jobs:
  setup-build-publish-deploy:
    name: 'Setup, Build, Publish, and Deploy'
    runs-on: 'ubuntu-latest'
    environment: 'production'

    permissions:
      contents: 'read'
      id-token: 'write'

    steps:
      - name: 'Checkout'
        uses: 'actions/checkout@692973e3d937129bcbf40652eb9f2f61becf3332' # actions/checkout@v4

      # Configure Workload Identity Federation and generate an access token.
      #
      # See https://github.com/google-github-actions/auth for more options,
      # including authenticating via a JSON credentials file.
      - id: 'auth'
        name: 'Authenticate to Google Cloud'
        uses: 'google-github-actions/auth@f112390a2df9932162083945e46d439060d66ec2' # google-github-actions/auth@v2
        with:
          workload_identity_provider: '${{ env.WORKLOAD_IDENTITY_PROVIDER }}'

      # Authenticate Docker to Google Cloud Artifact Registry
      - name: 'Docker Auth'
        uses: 'docker/login-action@9780b0c442fbb1117ed29e0efdff1e18412f7567' # docker/login-action@v3
        with:
          username: 'oauth2accesstoken'
          password: '${{ steps.auth.outputs.auth_token }}'
          registry: '${{ env.GAR_LOCATION }}-docker.pkg.dev'

      # Get the GKE credentials so we can deploy to the cluster
      - name: 'Set up GKE credentials'
        uses: 'google-github-actions/get-gke-credentials@6051de21ad50fbb1767bc93c11357a49082ad116' # google-github-actions/get-gke-credentials@v2
        with:
          cluster_name: '${{ env.GKE_CLUSTER }}'
          location: '${{ env.GKE_ZONE }}'

      # Build the Docker image
      - name: 'Build and push Docker container'
        run: |-
          DOCKER_TAG="${GAR_LOCATION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE}:${GITHUB_SHA}"

          docker build \
            --tag "${DOCKER_TAG}" \
            --build-arg GITHUB_SHA="${GITHUB_SHA}" \
            --build-arg GITHUB_REF="${GITHUB_REF}" \
            .

          docker push "${DOCKER_TAG}"

      # Set up kustomize
      - name: 'Set up Kustomize'
        run: |-
          curl -sfLo kustomize https://github.com/kubernetes-sigs/kustomize/releases/download/kustomize%2Fv5.4.3/kustomize_v5.4.3_linux_amd64.tar.gz
          chmod u+x ./kustomize

      # Deploy the Docker image to the GKE cluster
      - name: 'Deploy to GKE'
        run: |-
          # replacing the image name in the k8s template
          ./kustomize edit set image LOCATION-docker.pkg.dev/PROJECT_ID/REPOSITORY/IMAGE:TAG=$GAR_LOCATION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE:$GITHUB_SHA
          ./kustomize build . | kubectl apply -f -
          kubectl rollout status deployment/$DEPLOYMENT_NAME
          kubectl get services -o wide
