"""Dashboard exclusions for the employee product; native execution stays intact."""
from fastapi import HTTPException


def responsibility_authoring_only():
    raise HTTPException(status_code=410, detail="Author schedules in responsibility files. The Cron page is for execution and inspection.")


def employee_instructions_only():
    raise HTTPException(status_code=410, detail="Set employee.name and employee.instructions in the Config editor.")
