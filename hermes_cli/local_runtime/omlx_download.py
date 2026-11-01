"""Byte-progress bridge executed by the pinned oMLX Python, not Hermes Python."""

import json
import sys


def download(repo: str, destination: str) -> None:
    from huggingface_hub import snapshot_download
    from tqdm.auto import tqdm

    total = 0
    cached = 0

    def report(done: int) -> None:
        print("HERMES_OMLX_PROGRESS " + json.dumps([min(done, total), total]), flush=True)

    class Progress(tqdm):
        def __init__(self, *args, **kwargs):
            # The pinned Hub reports network and reconstruction bars; count model bytes once.
            self.model_bytes = kwargs.get("unit") == "B" and kwargs.get("desc", "").startswith("Reconstruct")
            self.transfer_bytes = kwargs.get("unit") == "B" and kwargs.get("desc", "").startswith("Downloading")
            kwargs["mininterval"] = 0.25
            kwargs["disable"] = False
            super().__init__(*args, **kwargs)

        def display(self, msg=None, pos=None):
            if self.model_bytes and total:
                report(cached + int(self.n))
            elif self.transfer_bytes and total:
                print("HERMES_OMLX_TRANSFER " + json.dumps(int(self.n)), flush=True)

    plan = snapshot_download(repo_id=repo, local_dir=destination, dry_run=True, tqdm_class=Progress)
    if not plan:
        raise ValueError("Repository contains no model files")
    total = sum(info.file_size for info in plan)
    cached = sum(info.file_size for info in plan if not info.will_download)
    report(cached)
    snapshot_download(repo_id=repo, local_dir=destination, revision=plan[0].commit_hash, tqdm_class=Progress)
    report(total)


if __name__ == "__main__":
    download(sys.argv[1], sys.argv[2])
