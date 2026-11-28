import shutil

def check(program: str) -> bool:
    return shutil.which(program) is not None
