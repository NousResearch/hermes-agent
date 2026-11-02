from neural.brain import Brain
from neural.memory import Memory
from neural.self_model import SelfModel


def test_neural_extension_points_have_stable_names():
    assert Brain().name == "brain"
    assert Memory().name == "memory"
    assert SelfModel().name == "self_model"
