from pydantic import BaseModel


class Session(BaseModel):
    id: str
    name: str
    status: str
