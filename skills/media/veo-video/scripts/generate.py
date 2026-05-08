import argparse
import json
import time
import urllib.request
import urllib.parse
import os
import uuid
import google.auth
from google.auth.transport.requests import Request

LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
MODEL = "veo-3.1-generate-001"

def get_auth():
    try:
        credentials, project_id = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        credentials.refresh(Request())
        return credentials.token, project_id
    except Exception as e:
        print(f"Error loading Google Cloud credentials: {e}")
        print("Please ensure GOOGLE_APPLICATION_CREDENTIALS is set to your service account key path.")
        exit(1)

def get_mime_type(filepath):
    ext = filepath.lower().split('.')[-1]
    if ext in ['jpg', 'jpeg']: return 'image/jpeg'
    if ext in ['png']: return 'image/png'
    if ext in ['mp4']: return 'video/mp4'
    return 'application/octet-stream'

def upload_to_gcs(local_path, token, bucket_name):
    if local_path.startswith("gs://"):
        return local_path # Already on GCS
    
    filename = os.path.basename(local_path)
    object_name = f"inputs/{uuid.uuid4().hex[:8]}_{filename}"
    mime_type = get_mime_type(local_path)
    
    url = f"https://storage.googleapis.com/upload/storage/v1/b/{bucket_name}/o?uploadType=media&name={urllib.parse.quote(object_name, safe='')}"
    with open(local_path, 'rb') as f:
        data = f.read()
        
    req = urllib.request.Request(url, method="POST", data=data)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", mime_type)
    
    print(f"Uploading {filename} to GCS bucket {bucket_name}...")
    with urllib.request.urlopen(req) as res:
        return f"gs://{bucket_name}/{object_name}"

def download_from_gcs(gcs_uri, local_path, token):
    path_parts = gcs_uri.replace("gs://", "").split("/", 1)
    bucket = path_parts[0]
    obj = urllib.parse.quote(path_parts[1], safe='')
    
    url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{obj}?alt=media"
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {token}")
    
    print(f"Downloading result to {local_path}...")
    with urllib.request.urlopen(req) as res, open(local_path, 'wb') as f:
        f.write(res.read())

def main():
    parser = argparse.ArgumentParser(description="Generate/Extend video using Vertex AI Veo 3.1")
    parser.add_argument("--prompt", required=True, help="Text prompt")
    parser.add_argument("--mode", choices=["text", "image", "first-last", "extend"], default="text")
    parser.add_argument("--image", help="Path/GCS URI to start frame image (required for image/first-last mode)")
    parser.add_argument("--last-frame", help="Path/GCS URI to last frame image (required for first-last mode)")
    parser.add_argument("--video", help="Path/GCS URI to source video (required for extend mode)")
    parser.add_argument("--aspect-ratio", choices=["16:9", "9:16"], default="16:9")
    parser.add_argument("--duration", choices=["4", "6", "8"], default="8")
    parser.add_argument("--output", default="output.mp4", help="Output file path")
    args = parser.parse_args()

    token, default_project_id = get_auth()
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or default_project_id
    
    if not project_id:
        print("Error: Could not determine Google Cloud Project ID. Set GOOGLE_CLOUD_PROJECT environment variable.")
        return
        
    bucket_name = os.getenv("VEO_GCS_BUCKET")
    if not bucket_name:
        print("Error: VEO_GCS_BUCKET environment variable must be set for uploading inputs and outputs.")
        return
    
    instance = {"prompt": args.prompt}
    
    if args.mode in ["image", "first-last"]:
        if not args.image:
            print("Error: --image is required for this mode.")
            return
        gcs_image = upload_to_gcs(args.image, token, bucket_name)
        instance["image"] = {"gcsUri": gcs_image, "mimeType": get_mime_type(args.image)}
        
    if args.mode == "first-last":
        if not args.last_frame:
            print("Error: --last-frame is required for first-last mode.")
            return
        gcs_last = upload_to_gcs(args.last_frame, token, bucket_name)
        instance["lastFrame"] = {"gcsUri": gcs_last, "mimeType": get_mime_type(args.last_frame)}
        
    if args.mode == "extend":
        if not args.video:
            print("Error: --video is required for extend mode.")
            return
        gcs_video = upload_to_gcs(args.video, token, bucket_name)
        instance["video"] = {"gcsUri": gcs_video, "mimeType": "video/mp4"}

    payload = {
        "instances": [instance],
        "parameters": {
            "aspectRatio": args.aspect_ratio,
            "durationSeconds": int(args.duration),
            "storageUri": f"gs://{bucket_name}/outputs"
        }
    }

    url_predict = f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{LOCATION}/publishers/google/models/{MODEL}:predictLongRunning"
    
    req = urllib.request.Request(url_predict, method="POST")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/json")

    print(f"Submitting {args.mode} task to Veo 3.1 (Project: {project_id}, Location: {LOCATION})...")
    try:
        with urllib.request.urlopen(req, data=json.dumps(payload).encode("utf-8")) as res:
            op_name = json.loads(res.read().decode())["name"]
            print(f"Operation started: {op_name}")
    except urllib.error.HTTPError as e:
        print(f"Failed to start: {e.read().decode()}")
        return

    url_fetch = f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{LOCATION}/publishers/google/models/{MODEL}:fetchPredictOperation"
    
    while True:
        req_poll = urllib.request.Request(url_fetch, method="POST")
        req_poll.add_header("Authorization", f"Bearer {token}")
        req_poll.add_header("Content-Type", "application/json")
        
        try:
            with urllib.request.urlopen(req_poll, data=json.dumps({"operationName": op_name}).encode("utf-8")) as res:
                poll_data = json.loads(res.read().decode())
                if poll_data.get("done"):
                    if "error" in poll_data:
                        print(f"Operation failed: {poll_data['error']}")
                        return
                    videos = poll_data.get("response", {}).get("videos", [])
                    if not videos:
                        print("Done, but no videos returned.")
                        return
                    
                    gcs_out = videos[0].get("gcsUri")
                    if gcs_out:
                        download_from_gcs(gcs_out, args.output, token)
                        print(f"Success! Video saved to {os.path.abspath(args.output)}")
                    else:
                        print("Done, but no gcsUri found.")
                    return
                else:
                    time.sleep(15)
        except urllib.error.HTTPError as e:
            print(f"Error polling: {e.read().decode()}")
            return

if __name__ == "__main__":
    main()